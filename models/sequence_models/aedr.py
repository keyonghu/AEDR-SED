import torch
import torch.nn as nn
import torch.nn.functional as F


class AcousticEdgeDrivenReset(nn.Module):
    """Causal acoustic-edge-driven temporal memory.

    Input and output tensors use ``[batch, channels, frames]``. ``reset_ema``
    is the stable AEDR formulation; ``legacy`` is retained for reproducing
    checkpoints trained with the original cumulative-average formulation.
    """

    def __init__(
        self,
        dim,
        mode="reset_ema",
        decay=0.95,
        kernel_size=15,
        residual_enabled=False,
        residual_scale_init=0.1,
    ):
        super().__init__()
        if mode not in {"legacy", "reset_ema"}:
            raise ValueError(f"Unsupported AEDR mode: {mode}")
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"AEDR decay must be in [0, 1), got {decay}")
        if kernel_size <= 0:
            raise ValueError(
                f"AEDR kernel_size must be a positive integer, got {kernel_size}"
            )
        if not 0.0 < residual_scale_init < 1.0:
            raise ValueError(
                "AEDR residual_scale_init must be strictly between 0 and 1, "
                f"got {residual_scale_init}"
            )

        self.mode = mode
        self.decay = float(decay)
        self.kernel_size = int(kernel_size)
        self.residual_enabled = bool(residual_enabled)
        self.local_history = nn.Conv1d(
            dim,
            dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size - 1,
            groups=dim,
        )
        self.infinite_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.edge_proj = nn.Sequential(
            nn.Conv1d(dim, max(1, dim // 4), kernel_size=1),
            nn.GELU(),
            nn.Conv1d(max(1, dim // 4), dim, kernel_size=1),
        )
        self.norm = nn.GroupNorm(1, dim) if mode == "legacy" else nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.apply(self._init_weights)
        if self.residual_enabled:
            initial_logit = torch.logit(torch.tensor(float(residual_scale_init)))
            self.residual_logit = nn.Parameter(initial_logit)
        else:
            self.register_parameter("residual_logit", None)

    @property
    def residual_scale(self):
        if self.residual_logit is None:
            return None
        return torch.sigmoid(self.residual_logit)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                "AEDR expects [batch, channels, frames], "
                f"received shape {tuple(x.shape)}"
            )

        _, _, sequence_length = x.shape
        x_shifted = F.pad(x[:, :, :-1], (1, 0), value=0.0)
        delta_x = torch.abs(x - x_shifted)
        reset_gate = torch.sigmoid(self.edge_proj(delta_x))

        local_mem = self.local_history(x)[:, :, :sequence_length]
        infinite_features = self.infinite_proj(x)

        if self.mode == "legacy":
            cumulative = torch.cumsum(infinite_features, dim=-1)
            time_steps = torch.arange(
                1,
                sequence_length + 1,
                device=x.device,
                dtype=infinite_features.dtype,
            ).view(1, 1, sequence_length)
            long_mem = cumulative / time_steps
        else:
            memory = torch.zeros_like(infinite_features[:, :, 0])
            memory_sequence = []
            for frame_index in range(sequence_length):
                candidate = infinite_features[:, :, frame_index]
                reset = reset_gate[:, :, frame_index]
                normal_update = self.decay * memory + (1.0 - self.decay) * candidate
                memory = (1.0 - reset) * normal_update + reset * candidate
                memory_sequence.append(memory)
            long_mem = torch.stack(memory_sequence, dim=-1)

        steady_mem = local_mem + long_mem
        fused = x * reset_gate + steady_mem * (1.0 - reset_gate)

        if self.mode == "legacy":
            transformed = self.act(self.norm(fused))
        else:
            fused = self.norm(fused.transpose(1, 2)).transpose(1, 2)
            transformed = self.act(fused)

        if not self.residual_enabled:
            return transformed

        # Blend temporal refinement with the aligned encoder features.
        scale = self.residual_scale.to(dtype=x.dtype, device=x.device)
        return x + scale * (transformed - x)
