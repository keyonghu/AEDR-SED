"""Align BEATs features, apply AEDR, and produce frame-level predictions."""

import os

import torch
from torch import nn
from torch.hub import download_url_to_file

from config import CHECKPOINT_URLS, RESOURCES_FOLDER
from models.sequence_models.aedr import AcousticEdgeDrivenReset


class PredictionsWrapper(nn.Module):
    """Align encoder tokens, refine them with AEDR, and apply linear heads."""

    def __init__(
        self,
        base_model,
        checkpoint=None,
        n_classes_strong=447,
        n_classes_weak=None,
        embed_dim=768,
        seq_len=250,
        aedr_enabled=True,
        aedr_mode="reset_ema",
        aedr_decay=0.95,
        aedr_kernel_size=15,
        aedr_residual=True,
        aedr_residual_scale_init=0.1,
    ):
        super().__init__()
        self.model = base_model
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.n_classes_strong = n_classes_strong
        self.n_classes_weak = (
            n_classes_strong if n_classes_weak is None else n_classes_weak
        )
        self.strong_head = nn.Linear(embed_dim, self.n_classes_strong)
        # The public initialization checkpoint includes both heads.
        self.weak_head = nn.Linear(embed_dim, self.n_classes_weak)

        if checkpoint is not None:
            print("Loading pretrained checkpoint:", checkpoint)
            self.load_checkpoint(checkpoint)

        # Initialize AEDR after loading the pretrained encoder and heads.
        self.aedr_enabled = bool(aedr_enabled)
        self.aedr = (
            AcousticEdgeDrivenReset(
                embed_dim,
                mode=aedr_mode,
                decay=aedr_decay,
                kernel_size=aedr_kernel_size,
                residual_enabled=aedr_residual,
                residual_scale_init=aedr_residual_scale_init,
            )
            if self.aedr_enabled
            else nn.Identity()
        )

    def load_checkpoint(self, checkpoint):
        """Load cached BEATs initialization, downloading it only if absent."""
        os.makedirs(RESOURCES_FOLDER, exist_ok=True)
        ckpt_file = os.path.join(RESOURCES_FOLDER, checkpoint + ".pt")
        if not os.path.exists(ckpt_file):
            download_url_to_file(CHECKPOINT_URLS[checkpoint], ckpt_file)
        state_dict = torch.load(ckpt_file, map_location="cpu", weights_only=True)

        # Public weights use model.model.* for the BEATs encoder.
        state_dict = {
            "model.beats." + key[len("model.model.") :]
            if key.startswith("model.model.")
            else key: value
            for key, value in state_dict.items()
        }
        current_state = self.state_dict()
        expected_missing = set()
        keys_to_remove = set()
        for keys in [
            ("weak_head.weight", "weak_head.bias"),
            ("strong_head.weight", "strong_head.bias"),
        ]:
            compatible = all(
                key in state_dict
                and key in current_state
                and state_dict[key].shape == current_state[key].shape
                for key in keys
            )
            if not compatible:
                keys_to_remove.update(key for key in keys if key in state_dict)
                expected_missing.update(key for key in keys if key in current_state)

        state_dict = {
            key: value for key, value in state_dict.items() if key not in keys_to_remove
        }
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if set(missing) != expected_missing or unexpected:
            raise RuntimeError(
                f"Checkpoint mismatch: missing={sorted(set(missing) - expected_missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

    def separate_params(self):
        return self.model.separate_params()

    def mel_forward(self, waveform):
        return self.model.mel_forward(waveform)

    def forward(self, features):
        sequence = self.model(features)
        if sequence.size(-2) > self.seq_len:
            sequence = torch.nn.functional.adaptive_avg_pool1d(
                sequence.transpose(1, 2), self.seq_len
            ).transpose(1, 2)
        elif sequence.size(-2) < self.seq_len:
            sequence = torch.nn.functional.interpolate(
                sequence.transpose(1, 2), size=self.seq_len, mode="linear"
            ).transpose(1, 2)
        refined = self.aedr(sequence.transpose(1, 2)).transpose(1, 2)
        strong = self.strong_head(refined)
        weak = self.weak_head(refined.mean(dim=1))
        return strong.transpose(1, 2), weak
