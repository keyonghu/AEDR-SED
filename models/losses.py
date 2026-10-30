import torch
import torch.nn as nn
import torch.nn.functional as F


def temporal_boundary_loss(
    pred_logits,
    targets,
    boundary_weight=10.0,
    mode="directional",
):
    """Penalize disagreement between predicted and target temporal edges."""
    if mode not in {"legacy", "directional"}:
        raise ValueError(f"Unsupported boundary loss mode: {mode}")

    if mode == "directional":
        probabilities = torch.sigmoid(pred_logits.float())
        targets = targets.float()
    else:
        probabilities = torch.sigmoid(pred_logits)

    pred_diff = probabilities[:, :, 1:] - probabilities[:, :, :-1]
    target_diff = targets[:, :, 1:] - targets[:, :, :-1]

    if mode == "legacy":
        return F.smooth_l1_loss(pred_diff.abs(), target_diff.abs(), reduction="mean")

    element_loss = F.smooth_l1_loss(pred_diff, target_diff, reduction="none")
    weights = 1.0 + float(boundary_weight) * target_diff.abs()
    return (element_loss * weights).sum() / weights.sum().clamp_min(1.0)


class OSFND(nn.Module):
    """Weight hard-label conflicts online; mixed targets retain unit weight."""

    def __init__(
        self,
        ema_alpha=0.95,
        initial_mu=0.3,
        initial_sigma=0.2,
        temperature=0.5,
        active_neg_threshold=0.1,
        neg_base_threshold=0.9,
        neg_temperature=0.2,
        pos_base_threshold=0.01,
        pos_temperature=0.2,
        min_weight=0.5,
    ):
        super().__init__()
        if temperature <= 0 or neg_temperature <= 0 or pos_temperature <= 0:
            raise ValueError("OSFND temperatures must be positive.")
        if not 0.0 <= ema_alpha < 1.0:
            raise ValueError("OSFND ema_alpha must be in [0, 1).")
        if not 0.0 <= min_weight <= 1.0:
            raise ValueError("OSFND min_weight must be in [0, 1].")

        self.ema_alpha = float(ema_alpha)
        self.temperature = float(temperature)
        self.active_neg_threshold = float(active_neg_threshold)
        self.neg_base_threshold = float(neg_base_threshold)
        self.neg_temperature = float(neg_temperature)
        self.pos_base_threshold = float(pos_base_threshold)
        self.pos_temperature = float(pos_temperature)
        self.min_weight = float(min_weight)
        self.register_buffer(
            "global_mu", torch.tensor(float(initial_mu), dtype=torch.float32)
        )
        self.register_buffer(
            "global_sigma",
            torch.tensor(float(initial_sigma), dtype=torch.float32),
        )

    @torch.no_grad()
    def forward(self, pred_logits, targets):
        logits = pred_logits.float()
        targets = targets.float()
        probabilities = torch.sigmoid(logits / self.temperature)
        smoothed = F.avg_pool1d(probabilities, kernel_size=5, stride=1, padding=2)

        hard_negative = targets <= 1e-6
        hard_positive = targets >= 1.0 - 1e-6
        active_negative = hard_negative & (probabilities > self.active_neg_threshold)
        active_probabilities = probabilities[active_negative]

        if active_probabilities.numel() > 0:
            batch_mu = active_probabilities.mean()
            batch_sigma = active_probabilities.std(unbiased=False)
            self.global_mu.mul_(self.ema_alpha).add_(batch_mu * (1.0 - self.ema_alpha))
            self.global_sigma.mul_(self.ema_alpha).add_(
                batch_sigma * (1.0 - self.ema_alpha)
            )

        dynamic_negative_threshold = torch.maximum(
            self.global_mu + self.global_sigma,
            self.global_mu.new_tensor(self.neg_base_threshold),
        )
        negative_excess = (smoothed - dynamic_negative_threshold).clamp_min(0.0)
        negative_weights = self.min_weight + (1.0 - self.min_weight) * torch.exp(
            -negative_excess / self.neg_temperature
        )

        positive_deficit = (self.pos_base_threshold - smoothed).clamp_min(0.0)
        positive_weights = self.min_weight + (1.0 - self.min_weight) * torch.exp(
            -positive_deficit / self.pos_temperature
        )

        weights = torch.ones_like(probabilities)
        weights = torch.where(hard_negative, negative_weights, weights)
        weights = torch.where(hard_positive, positive_weights, weights)

        hard_targets = hard_negative | hard_positive
        hard_target_count = hard_targets.sum().clamp_min(1)
        suppressed_ratio = (
            hard_targets & (weights < 0.8)
        ).sum().float() / hard_target_count.float()
        return weights, suppressed_ratio
