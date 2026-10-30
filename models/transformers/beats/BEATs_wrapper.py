import torch
from torch import nn

from models.transformers.beats.BEATs import BEATs, BEATsConfig


class BEATsWrapper(nn.Module):
    """Extract BEATs features with flattened-token or time-axis alignment."""

    VALID_ALIGNMENT_MODES = {"legacy_flatten", "temporal_mean"}

    def __init__(self, alignment_mode="legacy_flatten"):
        super().__init__()
        if alignment_mode not in self.VALID_ALIGNMENT_MODES:
            raise ValueError(
                f"Unknown BEATs alignment mode: {alignment_mode}. "
                f"Expected one of {sorted(self.VALID_ALIGNMENT_MODES)}."
            )
        cfg = BEATsConfig()
        self.beats = BEATs(cfg)
        self.alignment_mode = alignment_mode

    @staticmethod
    def _conv_output_size(size, kernel, stride, padding, dilation):
        return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

    def _temporal_frequency_mean(self, features, input_time, input_frequency):
        """Restore the time-frequency patch grid and average frequency patches."""
        patch = self.beats.patch_embedding
        kernel_t, kernel_f = patch.kernel_size
        stride_t, stride_f = patch.stride
        padding_t, padding_f = patch.padding
        dilation_t, dilation_f = patch.dilation
        time_patches = self._conv_output_size(
            input_time, kernel_t, stride_t, padding_t, dilation_t
        )
        frequency_patches = self._conv_output_size(
            input_frequency, kernel_f, stride_f, padding_f, dilation_f
        )
        expected_tokens = time_patches * frequency_patches
        if features.size(1) != expected_tokens:
            raise RuntimeError(
                "Cannot restore the BEATs time-frequency grid: "
                f"received {features.size(1)} tokens, but the patch embedding "
                f"geometry implies {time_patches} x {frequency_patches} = "
                f"{expected_tokens} tokens."
            )
        features = features.reshape(
            features.size(0), time_patches, frequency_patches, features.size(2)
        )
        return features.mean(dim=2)

    def mel_forward(self, x):
        with torch.autocast(device_type="cuda", enabled=False):
            mel = self.beats.preprocess(x)
        mel = mel.unsqueeze(1).transpose(2, 3)
        return mel

    def forward(self, x):
        x = x.transpose(2, 3)
        input_time, input_frequency = x.shape[-2:]
        features = self.beats.extract_features(x, do_preprocess=False)[0]
        if self.alignment_mode == "temporal_mean":
            features = self._temporal_frequency_mean(
                features, input_time, input_frequency
            )
        return features

    def separate_params(self):
        pt_params = [[], [], [], [], [], [], [], [], [], [], [], []]
        for k, p in self.named_parameters():
            if ".layers.0." in k:
                pt_params[0].append(p)
            elif ".layers.1." in k:
                pt_params[1].append(p)
            elif ".layers.2." in k:
                pt_params[2].append(p)
            elif ".layers.3." in k:
                pt_params[3].append(p)
            elif ".layers.4." in k:
                pt_params[4].append(p)
            elif ".layers.5." in k:
                pt_params[5].append(p)
            elif ".layers.6." in k:
                pt_params[6].append(p)
            elif ".layers.7." in k:
                pt_params[7].append(p)
            elif ".layers.8." in k:
                pt_params[8].append(p)
            elif ".layers.9." in k:
                pt_params[9].append(p)
            elif ".layers.10." in k:
                pt_params[10].append(p)
            elif ".layers.11." in k:
                pt_params[11].append(p)
            elif (
                ".post_extract_proj." in k
                or ".patch_embedding." in k
                or ".pos_conv." in k
                or "beats.layer_norm." in k
                or "beats.encoder.layer_norm." in k
            ):
                pt_params[0].append(p)
            else:
                raise ValueError(f"Check separate params for BEATs! Unknown key: {k}")
        return list(reversed(pt_params))
