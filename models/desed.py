"""Train and evaluate the shared AEDR-SED model on DESED soundscapes."""

import argparse
import os
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from config import DESED_ROOT
from data_utils import (
    LocalDesedStrongDataset,
    MixupDataset,
    get_uniform_sample_weights,
    get_weighted_sampler,
)
from models.class_names import DESED_SYNTHETIC_CLASSES
from models.engine import PLModule, get_dataloader_performance_kwargs
from models.training_config import (
    add_training_preset,
    apply_model_preset,
    print_training_config,
)
from utils import ManyHotEncoder, worker_init_fn


def resolve_desed_layout(dataset_root):
    """Resolve both common extraction layouts of the public archive."""
    requested = Path(dataset_root).expanduser().resolve()
    candidates = [
        requested,
        requested / "dcase_synth",
        requested / "dcase_synth" / "dcase_synth",
    ]
    required_relative_paths = {
        "train_audio_dir": Path("audio/train/synthetic21_train/soundscapes"),
        "train_annotation_file": Path(
            "metadata/train/synthetic21_train/soundscapes.tsv"
        ),
        "eval_audio_dir": Path("audio/validation/synthetic21_validation/soundscapes"),
        "eval_annotation_file": Path(
            "metadata/validation/synthetic21_validation/soundscapes.tsv"
        ),
    }
    for candidate in candidates:
        resolved = {
            name: candidate / relative
            for name, relative in required_relative_paths.items()
        }
        if all(path.exists() for path in resolved.values()):
            return candidate, resolved
    expected = "\n".join(
        f"  {relative}" for relative in required_relative_paths.values()
    )
    raise FileNotFoundError(
        f"Could not find the extracted DESED synthetic archive below "
        f"{requested}. Expected these relative paths:\n{expected}"
    )


def build_dataset(audio_dir, annotation_file, encoder):
    return LocalDesedStrongDataset(
        audio_dir=audio_dir,
        annotations_tsv=annotation_file,
        label_encoder=encoder,
    )


def make_loader(dataset, config, training):
    generator = torch.Generator()
    generator.manual_seed(config.random_seed + (0 if training else 1))
    kwargs = dict(
        dataset=dataset,
        worker_init_fn=worker_init_fn,
        generator=generator,
        batch_size=config.batch_size,
        **get_dataloader_performance_kwargs(config),
    )
    if training:
        epoch_len = min(config.epoch_len, len(dataset))
        if epoch_len != config.epoch_len:
            warnings.warn(
                f"epoch_len={config.epoch_len} exceeds the {len(dataset)} "
                f"available DESED files; using {epoch_len}."
            )
        sample_weights = get_uniform_sample_weights(dataset)
        kwargs.update(
            sampler=get_weighted_sampler(
                sample_weights,
                epoch_len=epoch_len,
                seed=config.random_seed,
            ),
            shuffle=False,
        )
    return DataLoader(**kwargs)


def train(config):
    encoder = ManyHotEncoder(DESED_SYNTHETIC_CLASSES)
    raw_train_set = build_dataset(
        config.train_audio_dir, config.train_annotation_file, encoder
    )
    train_set = (
        MixupDataset(raw_train_set, rate=config.wavmix_p)
        if config.wavmix_p > 0
        else raw_train_set
    )
    eval_set = build_dataset(
        config.eval_audio_dir, config.eval_annotation_file, encoder
    )
    train_dl = make_loader(train_set, config, training=True)
    eval_dl = make_loader(eval_set, config, training=False)

    run_dir = Path("logs")
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_logger = WandbLogger(
        project="AEDR-SED-DESED",
        notes=("AEDR-SED on the public DESED synthetic soundscape subset."),
        tags=["DESED synthetic", "strong SED", config.model_name],
        config=config,
        name=config.experiment_name,
        dir=run_dir,
    )
    csv_logger = CSVLogger(
        save_dir=run_dir,
        name="csv_metrics",
        version=config.experiment_name,
    )
    module = PLModule(config, encoder)
    callbacks = (
        [
            ModelCheckpoint(
                dirpath=Path("checkpoints") / config.experiment_name,
                # DESED follows the official DCASE scenario-1 PSDS, including
                # the across-class stability term (alpha_ST=1).
                monitor="val/psds1",
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
    if config.resume_checkpoint:
        print(
            f"Resuming complete training state from: {config.resume_checkpoint}",
            flush=True,
        )
    trainer.fit(
        module,
        train_dl,
        eval_dl,
        ckpt_path=config.resume_checkpoint,
    )
    wandb.finish()


def evaluate(config):
    encoder = ManyHotEncoder(DESED_SYNTHETIC_CLASSES)
    eval_set = build_dataset(
        config.eval_audio_dir, config.eval_annotation_file, encoder
    )
    eval_dl = make_loader(eval_set, config, training=False)
    if config.lightning_checkpoint:
        print(
            f"Loading Lightning checkpoint: {config.lightning_checkpoint}",
            flush=True,
        )
        module = PLModule.load_from_checkpoint(
            config.lightning_checkpoint,
            config=config,
            encoder=encoder,
        )
    else:
        module = PLModule(config, encoder)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=config.num_devices,
        precision=config.precision,
        num_sanity_val_steps=0,
        benchmark=not config.deterministic,
        deterministic=("warn" if config.deterministic else False),
    )
    trainer.validate(module, eval_dl)


def build_parser():
    parser = argparse.ArgumentParser(
        description=("Train or evaluate AEDR-SED on DESED synthetic soundscapes.")
    )
    add_training_preset(parser)
    parser.add_argument("--dataset_root", default=str(DESED_ROOT))
    parser.add_argument(
        "--eval_audio_dir",
        default=None,
        help=(
            "Optional strongly labelled DESED evaluation audio directory. "
            "When omitted, the synthetic21 validation split under "
            "--dataset_root is used."
        ),
    )
    parser.add_argument(
        "--eval_annotation_file",
        default=None,
        help=(
            "TSV paired with --eval_audio_dir. For the official real "
            "validation split this is usually metadata/validation/validation.tsv."
        ),
    )
    parser.add_argument("--experiment_name", default="aedr_sed_desed")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--random_seed", type=int, default=2024)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument(
        "--check_data_only",
        action="store_true",
        help="Validate paths, labels, resampling and tensor shapes, then exit.",
    )
    parser.add_argument(
        "--save_model", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--lightning_checkpoint")
    parser.add_argument("--resume_checkpoint")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)

    parser.set_defaults(model_name="BEATs")
    parser.add_argument(
        "--pretrained",
        choices=["scratch", "ssl", "weak", "strong"],
        default="weak",
    )
    parser.add_argument(
        "--beats_alignment_mode",
        choices=["legacy_flatten", "temporal_mean"],
        default="legacy_flatten",
    )

    # Retain the fixed training budget from the DESED experiment protocol.
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--epoch_len", type=int, default=10_000)
    parser.add_argument("--median_window", type=int, default=7)
    parser.add_argument("--compute_event_f1", action="store_true")
    parser.add_argument("--event_f1_threshold", type=float, default=0.5)
    parser.add_argument("--event_f1_thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--event_onset_collar", type=float, default=0.2)
    parser.add_argument("--event_offset_collar_rate", type=float, default=0.2)

    parser.add_argument(
        "--use_aedr", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--aedr_mode", choices=["legacy", "reset_ema"], default="reset_ema"
    )
    parser.add_argument("--aedr_decay", type=float, default=0.95)
    parser.add_argument("--aedr_kernel_size", type=int, default=15)
    parser.add_argument(
        "--aedr_residual", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--aedr_residual_scale_init", type=float, default=0.1)
    parser.add_argument(
        "--use_boundary_loss", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--boundary_alpha", type=float, default=0.05)
    parser.add_argument("--boundary_weight", type=float, default=10.0)
    parser.add_argument(
        "--boundary_mode",
        choices=["legacy", "directional"],
        default="directional",
    )
    parser.add_argument(
        "--use_osfnd", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--osfnd_warmup_epochs", type=int, default=30)
    parser.add_argument("--osfnd_ema_alpha", type=float, default=0.95)
    parser.add_argument("--osfnd_initial_mu", type=float, default=0.30)
    parser.add_argument("--osfnd_initial_sigma", type=float, default=0.20)
    parser.add_argument("--osfnd_temperature", type=float, default=0.50)
    parser.add_argument("--osfnd_active_neg_threshold", type=float, default=0.10)
    parser.add_argument("--osfnd_neg_base_threshold", type=float, default=0.90)
    parser.add_argument("--osfnd_neg_temperature", type=float, default=0.20)
    parser.add_argument("--osfnd_pos_base_threshold", type=float, default=0.01)
    parser.add_argument("--osfnd_pos_temperature", type=float, default=0.20)
    parser.add_argument("--osfnd_min_weight", type=float, default=0.50)

    parser.add_argument("--wavmix_p", type=float, default=0.5)
    parser.add_argument("--freq_warp_p", type=float, default=0.0)
    parser.add_argument("--filter_augment_p", type=float, default=0.0)
    parser.add_argument("--frame_shift_range", type=float, default=0.125)
    parser.add_argument("--mixup_p", type=float, default=0.3)
    parser.add_argument("--mixstyle_p", type=float, default=0.0)
    parser.add_argument("--max_time_mask_size", type=float, default=0.0)

    parser.add_argument("--adamw", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--schedule_mode", default="cos")
    parser.add_argument("--max_lr", type=float, default=1e-4)
    parser.add_argument("--seq_lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=float, default=0.0)
    parser.add_argument("--lr_end", type=float, default=2e-7)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    return parser


def validate_args(parser, args):
    try:
        apply_model_preset(args)
    except ValueError as error:
        parser.error(str(error))
    if (args.eval_audio_dir is None) != (args.eval_annotation_file is None):
        parser.error(
            "--eval_audio_dir and --eval_annotation_file must be provided together."
        )
    if args.random_seed < 0:
        parser.error("--random_seed must be non-negative.")
    if args.aedr_kernel_size <= 0:
        parser.error("--aedr_kernel_size must be positive.")
    if args.aedr_residual and not args.use_aedr:
        parser.error("--aedr_residual requires --use_aedr.")
    if not 0.0 < args.aedr_residual_scale_init < 1.0:
        parser.error("--aedr_residual_scale_init must be between zero and one.")
    if args.use_osfnd and args.osfnd_warmup_epochs >= args.n_epochs:
        parser.error("OSFND warmup must be smaller than --n_epochs.")
    if args.evaluate and not args.lightning_checkpoint:
        parser.error("--evaluate requires --lightning_checkpoint.")
    if args.check_data_only and (
        args.evaluate or args.lightning_checkpoint or args.resume_checkpoint
    ):
        parser.error(
            "--check_data_only cannot be combined with training/evaluation "
            "checkpoint options."
        )
    if args.evaluate and args.resume_checkpoint:
        parser.error("--resume_checkpoint is training-only.")
    if not args.evaluate and args.lightning_checkpoint:
        parser.error("--lightning_checkpoint is evaluation-only.")
    for checkpoint in [args.lightning_checkpoint, args.resume_checkpoint]:
        if checkpoint and not Path(checkpoint).is_file():
            parser.error(f"Checkpoint does not exist: {checkpoint}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    if args.print_config:
        print_training_config(args, "desed_synthetic")
        return
    resolved_root, paths = resolve_desed_layout(args.dataset_root)
    if args.eval_audio_dir is not None:
        explicit_audio = Path(args.eval_audio_dir).expanduser().resolve()
        explicit_annotations = Path(args.eval_annotation_file).expanduser().resolve()
        if not explicit_audio.is_dir():
            parser.error(
                f"DESED evaluation audio directory does not exist: {explicit_audio}"
            )
        if not explicit_annotations.is_file():
            parser.error(
                "DESED evaluation annotation file does not exist: "
                f"{explicit_annotations}"
            )
        paths["eval_audio_dir"] = explicit_audio
        paths["eval_annotation_file"] = explicit_annotations
        print(
            "Using explicitly supplied strongly labelled DESED evaluation split: "
            f"{explicit_audio}",
            flush=True,
        )
    for name, path in paths.items():
        setattr(args, name, str(path))
    print(f"Resolved DESED synthetic root: {resolved_root}", flush=True)

    # Fields consumed by the shared AudioSet module.
    args.dataset_name = "desed_synthetic"
    args.evaluation_classes = list(DESED_SYNTHETIC_CLASSES)
    args.mid_to_display_name_file = None
    # DCASE Task 4 PSDS scenario 1.
    args.psds_dtc_threshold = 0.7
    args.psds_gtc_threshold = 0.7
    # Cross-trigger evaluation is disabled in PSDS scenario 1.
    args.psds_cttc_threshold = None
    args.psds_alpha_ct = 0.0
    args.psds_alpha_st = 1.0
    args.psds_max_efpr = 100.0

    os.environ.setdefault("WANDB_MODE", "offline")
    pl.seed_everything(args.random_seed, workers=True)
    torch.backends.cudnn.benchmark = not args.deterministic
    if args.check_data_only:
        encoder = ManyHotEncoder(DESED_SYNTHETIC_CLASSES)
        for split_name, audio_dir, annotation_file in [
            (
                "train",
                args.train_audio_dir,
                args.train_annotation_file,
            ),
            (
                "validation",
                args.eval_audio_dir,
                args.eval_annotation_file,
            ),
        ]:
            dataset = build_dataset(audio_dir, annotation_file, encoder)
            sample = dataset[0]
            print(
                f"{split_name}: files={len(dataset)}, "
                f"audio_shape={tuple(sample['audio'].shape)}, "
                f"strong_shape={tuple(sample['strong'].shape)}, "
                f"duration={sample['duration']:.3f}s",
                flush=True,
            )
        print("DESED data check passed.", flush=True)
    elif args.evaluate:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
