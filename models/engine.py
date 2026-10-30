"""AEDR-SED model, optimization loop and AudioSet Strong evaluation."""

import argparse
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import sed_scores_eval
import torch
import torch.nn as nn
import transformers
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from data_utils import (
    LocalWavStrongDataset,
    MixupDataset,
    get_uniform_sample_weights,
    get_weighted_sampler,
)
from evaluation import compute_macro_event_metrics
from models.augmentation import (
    RandomResizeCrop,
    filter_augmentation,
    frame_shift,
    mixstyle,
    mixup,
    time_mask,
)
from models.class_names import as_strong_eval_classes, as_strong_train_classes
from models.losses import (
    OSFND,
    temporal_boundary_loss,
)
from models.prediction_wrapper import PredictionsWrapper
from models.training_config import (
    add_training_preset,
    apply_model_preset,
    check_audioset_paths,
    configure_audioset_paths,
    print_training_config,
)
from models.transformers.beats.BEATs_wrapper import BEATsWrapper
from utils import (
    ManyHotEncoder,
    batched_decode_preds,
    decode_score_dataframes_to_events,
    worker_init_fn,
)

torch.set_float32_matmul_precision("high")


class PLModule(pl.LightningModule):
    def __init__(self, config, encoder):
        super().__init__()

        self.config = config
        self.encoder = encoder
        n_classes = len(encoder.labels)

        checkpoint_names = {
            "scratch": None,
            "ssl": "BEATs_ssl",
            "weak": "BEATs_weak",
            "strong": "BEATs_strong_1",
        }
        beats_alignment_mode = config.beats_alignment_mode
        beats = BEATsWrapper(alignment_mode=beats_alignment_mode)
        print(f"BEATs SED alignment: {beats_alignment_mode}", flush=True)
        self.model = PredictionsWrapper(
            beats,
            checkpoint=checkpoint_names[config.pretrained],
            n_classes_strong=n_classes,
            n_classes_weak=n_classes,
            aedr_enabled=config.use_aedr,
            aedr_mode=config.aedr_mode,
            aedr_decay=config.aedr_decay,
            aedr_kernel_size=config.aedr_kernel_size,
            aedr_residual=config.aedr_residual,
            aedr_residual_scale_init=config.aedr_residual_scale_init,
        )

        self.strong_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.validation_loss = nn.BCEWithLogitsLoss()
        self.osfnd = (
            OSFND(
                ema_alpha=config.osfnd_ema_alpha,
                initial_mu=config.osfnd_initial_mu,
                initial_sigma=config.osfnd_initial_sigma,
                temperature=config.osfnd_temperature,
                active_neg_threshold=config.osfnd_active_neg_threshold,
                neg_base_threshold=config.osfnd_neg_base_threshold,
                neg_temperature=config.osfnd_neg_temperature,
                pos_base_threshold=config.osfnd_pos_base_threshold,
                pos_temperature=config.osfnd_pos_temperature,
                min_weight=config.osfnd_min_weight,
            )
            if config.use_osfnd
            else None
        )

        enabled_modules = [
            name
            for name, enabled in [
                ("AEDR", config.use_aedr),
                ("BoundaryLoss", config.use_boundary_loss),
                ("OSFND", config.use_osfnd),
            ]
            if enabled
        ]
        print("AEDR-SED: BEATs + " + " + ".join(enabled_modules))

        self.freq_warp = RandomResizeCrop((1, 1.0), time_scale=(1.0, 1.0))

        self.val_predictions_strong = {}
        self.val_ground_truth = {}
        self.val_duration = {}
        self.val_loss = []
        configured_event_thresholds = (
            config.event_f1_thresholds
            if config.event_f1_thresholds is not None
            else [config.event_f1_threshold]
        )
        self.event_f1_thresholds = sorted(
            set(
                [float(config.event_f1_threshold)]
                + [float(value) for value in configured_event_thresholds]
            )
        )

    def forward(self, batch):
        x = batch["audio"]
        mel = self.model.mel_forward(x)
        y_strong, _ = self.model(mel)
        return y_strong

    def get_optimizer(
        self,
        lr,
        seq_lr,
        lr_decay=0.0,
        adamw=False,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    ):
        # Use separate learning rates for the encoder and decoder/heads.
        if seq_lr is None:
            seq_lr = lr

        parameter_groups = []
        if lr_decay != 0.0:
            # BEATs groups are ordered from the output layers to the input.
            backbone_layers = self.model.model.separate_params()
            for layer_index, layer_params in enumerate(backbone_layers):
                parameter_groups.append(
                    {
                        "params": [p for p in layer_params if p.requires_grad],
                        "lr": lr * (lr_decay**layer_index),
                    }
                )
        else:
            parameter_groups.append(
                {
                    "params": [
                        param
                        for name, param in self.model.named_parameters()
                        if name.startswith("model.") and param.requires_grad
                    ],
                    "lr": lr,
                }
            )

        # AEDR and prediction heads use seq_lr.
        parameter_groups.append(
            {
                "params": [
                    param
                    for name, param in self.model.named_parameters()
                    if not name.startswith("model.") and param.requires_grad
                ],
                "lr": seq_lr,
            }
        )

        # Apply weight decay only to matrices and convolution kernels, not to
        # biases or normalization parameters.
        param_groups = []
        for group in parameter_groups:
            params_leq1d = [p for p in group["params"] if p.ndimension() <= 1]
            params_geq2d = [p for p in group["params"] if p.ndimension() >= 2]
            if params_leq1d:
                param_groups.append(
                    {
                        "params": params_leq1d,
                        "lr": group["lr"],
                        "weight_decay": 0.0,
                    }
                )
            if params_geq2d:
                param_groups.append(
                    {
                        "params": params_geq2d,
                        "lr": group["lr"],
                        "weight_decay": weight_decay,
                    }
                )

        if weight_decay > 0:
            assert adamw
        assert len(param_groups) > 0
        if adamw:
            print(f"\nUsing adamw weight_decay={weight_decay}!\n")
            return torch.optim.AdamW(param_groups, lr=lr, betas=betas)
        return torch.optim.Adam(param_groups, lr=lr, betas=betas)

    def get_lr_scheduler(
        self,
        optimizer,
        num_training_steps,
        schedule_mode="cos",
        gamma: float = 0.999996,
        num_warmup_steps=20000,
        lr_end=2e-7,
    ):
        if schedule_mode in {"exp"}:
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        if schedule_mode in {"cosine", "cos"}:
            return transformers.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        if schedule_mode in {"linear"}:
            print("Linear schedule!")
            return transformers.get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                power=1.0,
                lr_end=lr_end,
            )
        raise RuntimeError(f"schedule_mode={schedule_mode} Unknown.")

    def configure_optimizers(self):
        """Configure Adam/AdamW and the per-step learning-rate schedule."""
        optimizer = self.get_optimizer(
            self.config.max_lr,
            self.config.seq_lr,
            self.config.lr_decay,
            adamw=self.config.adamw,
            weight_decay=self.config.weight_decay,
        )

        num_training_steps = self.trainer.estimated_stepping_batches

        scheduler = self.get_lr_scheduler(
            optimizer,
            num_training_steps,
            schedule_mode=self.config.schedule_mode,
            num_warmup_steps=self.config.warmup_steps,
            lr_end=self.config.lr_end,
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler_config]

    def training_step(self, train_batch, batch_idx):
        """Optimize reliability-weighted BCE and the directional boundary term."""

        x = train_batch["audio"]
        labels = train_batch["strong"]
        mel = self.model.mel_forward(x)

        if self.config.frame_shift_range > 0:
            mel, labels = frame_shift(
                mel,
                labels,
                net_pooling=self.encoder.net_pooling,
                shift_range=self.config.frame_shift_range,
            )

        if self.config.mixup_p > random.random():
            mel, labels = mixup(
                mel,
                targets=labels,
            )

        if self.config.mixstyle_p > random.random():
            mel = mixstyle(mel)

        if self.config.max_time_mask_size > 0:
            mel, labels = time_mask(
                mel,
                labels,
                net_pooling=self.encoder.net_pooling,
                max_mask_ratio=self.config.max_time_mask_size,
            )

        if self.config.filter_augment_p > random.random():
            mel, _ = filter_augmentation(mel)

        if self.config.freq_warp_p > random.random():
            mel = mel.squeeze(1)
            mel = self.freq_warp(mel)
            mel = mel.unsqueeze(1)

        y_hat_strong, _ = self.model(mel)

        supervised_element_loss = self.strong_loss(y_hat_strong, labels)
        raw_classification_loss = supervised_element_loss.mean()
        osfnd_weights = torch.ones_like(supervised_element_loss)

        if (
            self.config.use_osfnd
            and self.current_epoch >= self.config.osfnd_warmup_epochs
        ):
            osfnd_weights, suppressed_ratio = self.osfnd(y_hat_strong, labels)
            supervised_element_loss = supervised_element_loss * osfnd_weights
        else:
            suppressed_ratio = y_hat_strong.new_zeros(())

        classification_loss = supervised_element_loss.mean()
        negative_mask = labels <= 0.5
        positive_mask = ~negative_mask
        negative_weight_mean = (
            osfnd_weights[negative_mask].mean()
            if negative_mask.any()
            else y_hat_strong.new_ones(())
        )
        positive_weight_mean = (
            osfnd_weights[positive_mask].mean()
            if positive_mask.any()
            else y_hat_strong.new_ones(())
        )
        loss_reduction_ratio = (
            1.0 - classification_loss / raw_classification_loss.clamp_min(1e-8)
        )
        if self.config.use_boundary_loss:
            boundary_loss = temporal_boundary_loss(
                y_hat_strong,
                labels,
                boundary_weight=self.config.boundary_weight,
                mode=self.config.boundary_mode,
            )
        else:
            boundary_loss = y_hat_strong.new_zeros(())

        strong_supervised_loss = (
            classification_loss + self.config.boundary_alpha * boundary_loss
        )

        loss = strong_supervised_loss

        self.log("epoch", self.current_epoch)
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(f"trainer/lr_optimizer_{i}", param_group["lr"])
        self.log("train/loss", loss.detach().cpu(), prog_bar=True)
        self.log("train/strong_supervised_loss", strong_supervised_loss.detach().cpu())
        self.log(
            "train/classification_loss_raw", raw_classification_loss.detach().cpu()
        )
        self.log(
            "train/classification_loss_weighted", classification_loss.detach().cpu()
        )
        self.log("train/classification_loss", classification_loss.detach().cpu())
        self.log("train/boundary_loss", boundary_loss.detach().cpu())
        self.log("train/osfnd_suppressed_ratio", suppressed_ratio.detach().cpu())
        self.log("train/osfnd_weight_mean", osfnd_weights.mean().detach().cpu())
        self.log(
            "train/osfnd_negative_weight_mean", negative_weight_mean.detach().cpu()
        )
        self.log(
            "train/osfnd_positive_weight_mean", positive_weight_mean.detach().cpu()
        )
        self.log(
            "train/osfnd_loss_reduction_ratio", loss_reduction_ratio.detach().cpu()
        )
        if self.osfnd is not None:
            self.log("train/osfnd_global_mu", self.osfnd.global_mu.detach().cpu())
            self.log("train/osfnd_global_sigma", self.osfnd.global_sigma.detach().cpu())

        return loss

    def validation_step(self, val_batch, batch_idx):
        durations = val_batch["duration"]
        for sample_idx, (f, gt_string) in enumerate(
            zip(val_batch["filename"], val_batch["gt_string"])
        ):
            f = Path(f).stem
            events = [event.split(";;") for event in gt_string.split("++") if event]
            self.val_ground_truth[f] = [
                (float(event[0]), float(event[1]), event[2]) for event in events
            ]
            self.val_duration[f] = float(durations[sample_idx])

        y_hat_strong = self(val_batch)
        y_strong = val_batch["strong"]

        loss = self.validation_loss(y_hat_strong, y_strong)
        self.val_loss.append(loss.cpu())

        scores_raw, scores_postprocessed, prediction_dfs = batched_decode_preds(
            y_hat_strong.float(),
            val_batch["filename"],
            self.encoder,
            median_filter=self.config.median_window,
        )

        self.val_predictions_strong.update(scores_postprocessed)

    def on_validation_epoch_end(self):
        gt_unique_events = set(
            [e[2] for f, events in self.val_ground_truth.items() for e in events]
        )
        train_unique_events = set(self.encoder.labels)
        # Evaluate classes shared by the model and the annotated subset.
        class_intersection = gt_unique_events.intersection(train_unique_events)

        if self.config.evaluation_classes is None:
            expected_class_intersection = set(as_strong_train_classes).intersection(
                as_strong_eval_classes
            )
        else:
            expected_class_intersection = set(
                self.config.evaluation_classes
            ).intersection(train_unique_events)

        if class_intersection != expected_class_intersection:
            warnings.warn(
                "Evaluation covers "
                f"{len(class_intersection)} of the expected "
                f"{len(expected_class_intersection)} supported classes."
            )

        val_ground_truth = {
            fid: [
                event
                for event in self.val_ground_truth[fid]
                if event[2] in class_intersection
            ]
            for fid in self.val_ground_truth
        }
        # drop audios without events - aligned with DESED evaluation procedure
        val_ground_truth = {
            fid: events for fid, events in val_ground_truth.items() if len(events) > 0
        }
        audio_durations = {
            fid: self.val_duration[fid] for fid in val_ground_truth.keys()
        }

        as_strong_preds = {
            fid: self.val_predictions_strong[fid] for fid in val_ground_truth.keys()
        }
        unused_classes = list(set(self.encoder.labels).difference(class_intersection))
        for f, df in as_strong_preds.items():
            df.drop(columns=list(unused_classes), axis=1, inplace=True)

        segment_based_pauroc = sed_scores_eval.segment_based.auroc(
            as_strong_preds,
            val_ground_truth,
            audio_durations,
            max_fpr=0.1,
            segment_length=1.0,
            num_jobs=1,
        )

        psds1 = sed_scores_eval.intersection_based.psds(
            as_strong_preds,
            val_ground_truth,
            audio_durations,
            dtc_threshold=self.config.psds_dtc_threshold,
            gtc_threshold=self.config.psds_gtc_threshold,
            cttc_threshold=self.config.psds_cttc_threshold,
            alpha_ct=self.config.psds_alpha_ct,
            alpha_st=self.config.psds_alpha_st,
            max_efpr=self.config.psds_max_efpr,
            num_jobs=1,
        )

        # Macro PSDS1 is the mean of the class-wise scores.
        val_loss = torch.as_tensor(self.val_loss).mean().to(self.device)
        psds1_value = float(psds1[0])
        psds1_macro_averaged = float(np.array([v for k, v in psds1[1].items()]).mean())
        pauroc = float(segment_based_pauroc[0]["mean"])

        event_metrics = {}
        event_metrics_by_threshold = {}
        if self.config.compute_event_f1:
            ordered_event_classes = [
                label for label in self.encoder.labels if label in class_intersection
            ]
            event_encoder = ManyHotEncoder(
                ordered_event_classes,
                audio_len=self.encoder.audio_len,
                frame_hop=self.encoder.frame_hop,
                net_pooling=self.encoder.net_pooling,
                fs=self.encoder.fs,
            )
            for threshold in self.event_f1_thresholds:
                event_predictions = decode_score_dataframes_to_events(
                    score_dataframes=as_strong_preds,
                    encoder=event_encoder,
                    threshold=threshold,
                    scores_are_logits=True,
                )
                event_metrics_by_threshold[threshold] = compute_macro_event_metrics(
                    predictions=event_predictions,
                    ground_truth=val_ground_truth,
                    event_classes=ordered_event_classes,
                    onset_collar=self.config.event_onset_collar,
                    offset_collar_rate=self.config.event_offset_collar_rate,
                )
            event_metrics = event_metrics_by_threshold[
                float(self.config.event_f1_threshold)
            ]

        logs = {
            "val/loss": val_loss,
            "val/psds1": psds1_value,
            "val/psds1_macro_averaged": psds1_macro_averaged,
            "val/pauroc": pauroc,
        }
        if event_metrics:
            logs.update(
                {
                    "val/event_precision": event_metrics["event_precision"],
                    "val/event_recall": event_metrics["event_recall"],
                    "val/event_f1": event_metrics["event_f1"],
                    "val/event_precision_micro": event_metrics["event_precision_micro"],
                    "val/event_recall_micro": event_metrics["event_recall_micro"],
                    "val/event_f1_micro": event_metrics["event_f1_micro"],
                    "val/event_predicted_class_count": event_metrics[
                        "predicted_class_count"
                    ],
                    "val/event_nonzero_f1_class_count": event_metrics[
                        "nonzero_f1_class_count"
                    ],
                }
            )
            for threshold, threshold_metrics in event_metrics_by_threshold.items():
                threshold_tag = f"{threshold:.2f}".replace(".", "p")
                logs.update(
                    {
                        f"val/event_f1_t{threshold_tag}": threshold_metrics["event_f1"],
                        f"val/event_f1_micro_t{threshold_tag}": threshold_metrics[
                            "event_f1_micro"
                        ],
                    }
                )

        self.log_dict(logs, sync_dist=False, prog_bar=True)
        if self.trainer.is_global_zero:
            metrics_message = (
                "\nValidation metrics "
                f"(epoch {self.current_epoch}): "
                f"loss={float(val_loss):.6f}, "
                f"psds1={psds1_value:.6f}, "
                f"psds1_macro_averaged={psds1_macro_averaged:.6f}, "
                f"pauroc={pauroc:.6f}"
            )
            if event_metrics:
                metrics_message += (
                    f", event_precision={event_metrics['event_precision']:.6f}"
                    f", event_recall={event_metrics['event_recall']:.6f}"
                    f", event_f1={event_metrics['event_f1']:.6f}"
                    f" @ threshold={self.config.event_f1_threshold:.2f}"
                )
            print(metrics_message, flush=True)
            if len(event_metrics_by_threshold) > 1:
                print(
                    "\nEvent-F1 threshold diagnostic "
                    f"(median_window={self.config.median_window}, "
                    f"classes={int(event_metrics['evaluated_class_count'])}, "
                    f"reference_events={int(event_metrics['reference_event_count'])}):"
                )
                print(
                    "threshold  macro_P  macro_R  macro_F1  micro_F1  "
                    "pred_events  pred_classes  nonzero_classes"
                )
                for threshold, threshold_metrics in event_metrics_by_threshold.items():
                    print(
                        f"{threshold:>9.2f}  "
                        f"{threshold_metrics['event_precision']:>7.4f}  "
                        f"{threshold_metrics['event_recall']:>7.4f}  "
                        f"{threshold_metrics['event_f1']:>8.4f}  "
                        f"{threshold_metrics['event_f1_micro']:>8.4f}  "
                        f"{int(threshold_metrics['prediction_event_count']):>11d}  "
                        f"{int(threshold_metrics['predicted_class_count']):>12d}  "
                        f"{int(threshold_metrics['nonzero_f1_class_count']):>15d}"
                    )
        self.val_predictions_strong = {}
        self.val_ground_truth = {}
        self.val_duration = {}
        self.val_loss = []


def get_configured_eval_dataset(config, encoder):
    return LocalWavStrongDataset(
        audio_dir=config.eval_audio_dir,
        annotations_tsv=config.eval_annotation_file,
        mid_to_display_name_tsv=config.mid_to_display_name_file,
        label_encoder=encoder,
    )


def get_configured_training_dataset(config, encoder):
    dataset = LocalWavStrongDataset(
        audio_dir=config.train_audio_dir,
        annotations_tsv=config.train_annotation_file,
        mid_to_display_name_tsv=config.mid_to_display_name_file,
        label_encoder=encoder,
    )
    return (
        MixupDataset(dataset, rate=config.wavmix_p) if config.wavmix_p > 0 else dataset
    )


def get_dataloader_performance_kwargs(config):
    kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if config.num_workers > 0:
        kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": config.prefetch_factor,
            }
        )
    return kwargs


def train(config):
    """Fine-tune AEDR-SED using development-set checkpoint selection."""

    run_dir = Path("logs")
    run_dir.mkdir(parents=True, exist_ok=True)

    wandb_logger = WandbLogger(
        project="AEDR-SED",
        notes="AEDR-SED temporal modeling and reliability-aware supervision on AudioSet Strong.",
        tags=[
            "AEDR-SED",
            "AudioSet Strong",
            "Sound Event Detection",
            config.model_name,
        ],
        config=config,
        name=config.experiment_name,
        dir=run_dir,
    )
    csv_logger = CSVLogger(
        save_dir=run_dir,
        name="csv_metrics",
        version=config.experiment_name,
    )

    encoder = ManyHotEncoder(as_strong_train_classes)

    train_set = get_configured_training_dataset(config, encoder)
    eval_set = get_configured_eval_dataset(config, encoder)

    sample_weights = get_uniform_sample_weights(train_set)

    epoch_len = config.epoch_len
    if config.train_audio_dir is not None and epoch_len > len(train_set):
        warnings.warn(
            f"epoch_len={epoch_len} exceeds the {len(train_set)} available local "
            f"training files. Using epoch_len={len(train_set)}."
        )
        epoch_len = len(train_set)
    train_sampler = get_weighted_sampler(
        sample_weights,
        epoch_len=epoch_len,
        seed=config.random_seed,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(config.random_seed)
    eval_generator = torch.Generator()
    eval_generator.manual_seed(config.random_seed + 1)

    train_dl = DataLoader(
        dataset=train_set,
        sampler=train_sampler,
        worker_init_fn=worker_init_fn,
        generator=train_generator,
        batch_size=config.batch_size,
        shuffle=False,
        **get_dataloader_performance_kwargs(config),
    )

    eval_dl = DataLoader(
        dataset=eval_set,
        worker_init_fn=worker_init_fn,
        generator=eval_generator,
        batch_size=config.batch_size,
        **get_dataloader_performance_kwargs(config),
    )

    pl_module = PLModule(config, encoder)

    callbacks = (
        [
            ModelCheckpoint(
                dirpath=Path("checkpoints") / config.experiment_name,
                monitor="val/psds1_macro_averaged",
                mode="max",
                save_top_k=1,
                save_last=True,
            )
        ]
        if config.save_model
        else []
    )

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        logger=[wandb_logger, csv_logger],
        accelerator="auto",
        devices=config.num_devices,
        precision=config.precision,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        accumulate_grad_batches=config.accumulate_grad_batches,
        benchmark=not config.deterministic,
        deterministic=("warn" if config.deterministic else False),
        callbacks=callbacks,
    )

    # Restore optimizer, scheduler and epoch state when resuming.
    if config.resume_checkpoint is not None:
        print(
            f"Resuming complete training state from: {config.resume_checkpoint}",
            flush=True,
        )
    trainer.fit(
        pl_module,
        train_dl,
        eval_dl,
        ckpt_path=config.resume_checkpoint,
    )

    wandb.finish()


def evaluate(config):
    encoder = ManyHotEncoder(as_strong_train_classes)
    eval_set = get_configured_eval_dataset(config, encoder)

    eval_generator = torch.Generator()
    eval_generator.manual_seed(config.random_seed + 1)
    eval_dl = DataLoader(
        dataset=eval_set,
        worker_init_fn=worker_init_fn,
        generator=eval_generator,
        batch_size=config.batch_size,
        **get_dataloader_performance_kwargs(config),
    )

    if config.lightning_checkpoint is not None:
        print(f"Loading Lightning checkpoint: {config.lightning_checkpoint}")
        pl_module = PLModule.load_from_checkpoint(
            config.lightning_checkpoint,
            config=config,
            encoder=encoder,
        )
    else:
        pl_module = PLModule(config, encoder)

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator="auto",
        devices=config.num_devices,
        precision=config.precision,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        accumulate_grad_batches=config.accumulate_grad_batches,
        benchmark=not config.deterministic,
        deterministic=("warn" if config.deterministic else False),
    )

    trainer.validate(pl_module, eval_dl)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train or evaluate AEDR-SED on AudioSet Strong."
    )
    add_training_preset(parser)

    # general
    parser.add_argument("--experiment_name", type=str, default="aedr_sed_audioset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument(
        "--random_seed",
        type=int,
        default=2024,
        help=(
            "Global seed for model initialization, sampling, data-loader "
            "workers and stochastic augmentation. Use a different explicit "
            "value for every repeated-seed run."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help=(
            "Request deterministic PyTorch/Lightning kernels. This improves "
            "repeatability for paper runs but can reduce throughput."
        ),
    )
    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument(
        "--save_model", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--lightning_checkpoint",
        type=str,
        default=None,
        help="Lightning .ckpt file to evaluate with --evaluate.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help=(
            "Lightning .ckpt used to resume training state, including the "
            "optimizer, scheduler, epoch and global step. n_epochs is the "
            "total target epoch count, not the number of additional epochs."
        ),
    )
    parser.add_argument("--train_audio_dir", type=str, default=None)
    parser.add_argument("--train_annotation_file", type=str, default=None)
    parser.add_argument("--eval_audio_dir", type=str, default=None)
    parser.add_argument("--eval_annotation_file", type=str, default=None)
    parser.add_argument("--mid_to_display_name_file", type=str, default=None)

    # model
    parser.set_defaults(model_name="BEATs")
    parser.add_argument(
        "--beats_alignment_mode",
        type=str,
        choices=["legacy_flatten", "temporal_mean"],
        default="legacy_flatten",
        help=(
            "BEATs-to-SED alignment. temporal_mean restores the time-frequency "
            "patch grid, averages the eight frequency patches, and lets the "
            "prediction wrapper interpolate the pure temporal sequence to "
            "250 frames. legacy_flatten matches the AEDR-SED paper setting."
        ),
    )
    # "scratch" = no pretraining
    # "ssl" = SSL pre-trained
    # "weak" = AudioSet Weak pre-trained
    # "strong" = AudioSet Strong pre-trained
    parser.add_argument(
        "--pretrained",
        type=str,
        choices=["scratch", "ssl", "weak", "strong"],
        default="weak",
    )

    # training
    parser.add_argument("--n_epochs", type=int, default=120)
    parser.add_argument("--epoch_len", type=int, default=20_000)
    parser.add_argument("--median_window", type=int, default=7)
    parser.add_argument(
        "--compute_event_f1",
        action="store_true",
        default=False,
        help=(
            "Compute dataset-level macro event precision, recall and F1 using "
            "sed_eval. The existing PSDS1 calculation is unchanged."
        ),
    )
    parser.add_argument(
        "--event_f1_threshold",
        type=float,
        default=0.5,
        help="Fixed sigmoid-probability threshold for Event-based F1.",
    )
    parser.add_argument(
        "--event_f1_thresholds",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional diagnostic threshold list evaluated after one model "
            "inference pass. No best threshold is selected automatically."
        ),
    )
    parser.add_argument(
        "--event_onset_collar",
        type=float,
        default=0.2,
        help="Onset collar in seconds; also the minimum offset collar.",
    )
    parser.add_argument(
        "--event_offset_collar_rate",
        type=float,
        default=0.2,
        help="Allowed offset error as a fraction of reference-event duration.",
    )

    # Component switches inherit the selected model preset.
    parser.add_argument(
        "--use_aedr", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--aedr_mode", type=str, choices=["legacy", "reset_ema"], default="reset_ema"
    )
    parser.add_argument("--aedr_decay", type=float, default=0.95)
    parser.add_argument("--aedr_kernel_size", type=int, default=15)
    parser.add_argument(
        "--aedr_residual",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Blend the original temporal features with AEDR through a "
            "learnable bounded residual scale."
        ),
    )
    parser.add_argument("--aedr_residual_scale_init", type=float, default=0.1)

    parser.add_argument(
        "--use_boundary_loss", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--boundary_alpha", type=float, default=0.05)
    parser.add_argument("--boundary_weight", type=float, default=10.0)
    parser.add_argument(
        "--boundary_mode",
        type=str,
        choices=["legacy", "directional"],
        default="directional",
    )

    parser.add_argument(
        "--use_osfnd", action=argparse.BooleanOptionalAction, default=None
    )
    # OSFND reference settings.
    parser.add_argument("--osfnd_warmup_epochs", type=int, default=30)
    parser.add_argument("--osfnd_ema_alpha", type=float, default=0.95)
    parser.add_argument("--osfnd_initial_mu", type=float, default=0.30)
    parser.add_argument("--osfnd_initial_sigma", type=float, default=0.20)
    parser.add_argument("--osfnd_temperature", type=float, default=0.5)
    parser.add_argument("--osfnd_active_neg_threshold", type=float, default=0.1)
    parser.add_argument("--osfnd_neg_base_threshold", type=float, default=0.90)
    parser.add_argument("--osfnd_neg_temperature", type=float, default=0.20)
    parser.add_argument("--osfnd_pos_base_threshold", type=float, default=0.01)
    parser.add_argument("--osfnd_pos_temperature", type=float, default=0.20)
    parser.add_argument("--osfnd_min_weight", type=float, default=0.50)

    # augmentation
    parser.add_argument("--wavmix_p", type=float, default=0.5)
    parser.add_argument("--freq_warp_p", type=float, default=0.0)
    parser.add_argument("--filter_augment_p", type=float, default=0.0)
    parser.add_argument("--frame_shift_range", type=float, default=0.125)  # in seconds
    parser.add_argument("--mixup_p", type=float, default=0.3)
    parser.add_argument("--mixstyle_p", type=float, default=0.0)
    parser.add_argument("--max_time_mask_size", type=float, default=0.0)

    # optimizer
    parser.add_argument("--adamw", action="store_true", default=False)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # lr schedule
    parser.add_argument("--schedule_mode", type=str, default="cos")

    # Backbone and temporal-module learning rates for the paper protocol.
    parser.add_argument("--max_lr", type=float, default=1e-4)
    parser.add_argument("--seq_lr", type=float, default=1e-4)

    parser.add_argument("--lr_decay", type=float, default=0.0)
    parser.add_argument("--lr_end", type=float, default=2e-7)
    parser.add_argument("--warmup_steps", type=int, default=1000)

    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        apply_model_preset(args)
        configure_audioset_paths(args)
    except ValueError as error:
        parser.error(str(error))

    if args.aedr_kernel_size <= 0:
        parser.error("--aedr_kernel_size must be a positive integer.")
    if not 0.0 < args.aedr_residual_scale_init < 1.0:
        parser.error("--aedr_residual_scale_init must be strictly between 0 and 1.")
    if args.aedr_residual and not args.use_aedr:
        parser.error("--aedr_residual requires --use_aedr.")
    if args.use_osfnd and args.osfnd_warmup_epochs >= args.n_epochs:
        warnings.warn(
            "OSFND will never activate because osfnd_warmup_epochs is not "
            "smaller than n_epochs."
        )
    if not 0.0 < args.event_f1_threshold < 1.0:
        parser.error("--event_f1_threshold must be strictly between 0 and 1.")
    if args.event_f1_thresholds is not None and any(
        not 0.0 < threshold < 1.0 for threshold in args.event_f1_thresholds
    ):
        parser.error(
            "Every value in --event_f1_thresholds must be strictly between 0 and 1."
        )
    if args.event_onset_collar < 0.0:
        parser.error("--event_onset_collar must be non-negative.")
    if args.event_offset_collar_rate < 0.0:
        parser.error("--event_offset_collar_rate must be non-negative.")
    if args.random_seed < 0:
        parser.error("--random_seed must be non-negative.")
    if (
        args.evaluate
        and args.lightning_checkpoint is None
        and any(
            [
                args.use_aedr,
                args.use_osfnd,
                args.use_boundary_loss,
            ]
        )
    ):
        parser.error(
            "AEDR-SED evaluation requires --lightning_checkpoint. Public encoder "
            "weights alone are not a trained AEDR-SED model."
        )
    return args


def main():
    parser = build_parser()
    args = parse_args()
    if args.print_config:
        print_training_config(args, "audioset_strong")
        return
    try:
        check_audioset_paths(args)
    except ValueError as error:
        build_parser().error(str(error))
    # Use offline logging unless online logging is explicitly enabled.
    os.environ.setdefault("WANDB_MODE", "offline")
    pl.seed_everything(args.random_seed, workers=True)
    torch.backends.cudnn.benchmark = not args.deterministic

    if args.evaluate:
        if args.resume_checkpoint is not None:
            parser.error(
                "--resume_checkpoint is training-only and cannot be used "
                "together with --evaluate."
            )
        if args.lightning_checkpoint is None:
            assert args.pretrained == "strong", (
                "Without --lightning_checkpoint, evaluation requires an AudioSet Strong pretrained model."
            )
        evaluate(args)
    else:
        if args.lightning_checkpoint is not None:
            raise ValueError(
                "--lightning_checkpoint is only supported together with --evaluate."
            )
        if (
            args.resume_checkpoint is not None
            and not Path(args.resume_checkpoint).is_file()
        ):
            parser.error(f"Resume checkpoint does not exist: {args.resume_checkpoint}")
        train(args)


if __name__ == "__main__":
    main()
