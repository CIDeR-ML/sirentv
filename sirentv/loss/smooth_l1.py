from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from sirentv.loss.builder import LOSSES
from sirentv.utils.misc import predict_gradient_magnitudes

@LOSSES.register_module()
class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, key: str, weight=1.0, reduce_method="mean", **kwargs):
        super().__init__()
        assert reduce_method in ["mean", "sum", "none"], (
            f"Invalid reduction method: {reduce_method}"
        )
        self.reduce = (
            getattr(torch, reduce_method) if reduce_method != "none" else lambda x: x
        )
        self.key = key
        self.weight = weight
        self.kwargs = kwargs

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        weight: dict[str, torch.Tensor] | None = None,
    ):
        if weight is None:
            weight = {self.key: torch.ones_like(pred[self.key])}

        device = pred[self.key].device
        weight[self.key] = weight[self.key].to(device)
        target[self.key] = target[self.key].to(device)
        loss = weight[self.key] * F.smooth_l1_loss(pred[self.key], target[self.key], **self.kwargs)
        return self.weight * self.reduce(loss)


@LOSSES.register_module()
class SmoothL1Loss(WeightedSmoothL1Loss):
    def __init__(self, key: str, weight=1.0, reduce_method="mean", **kwargs):
        super().__init__(key, weight, reduce_method)
    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        weight: dict[str, torch.Tensor] | None = None,
    ):
        return super().forward(pred, target, weight=None, **self.kwargs)

@LOSSES.register_module()
class VisibilityGradient_SmoothL1Loss(nn.Module):
    """Gradient magnitude loss using any base loss function"""

    def __init__(self, key: str = 'v', weight=1.0, reduce_method="mean", threshold=1.0E-9, **kwargs):
        super().__init__()
        self.key = key
        self.weight = weight
        self.reduce_method = reduce_method
        self.threshold = threshold

        # Running statistics for normalization
        self.register_buffer('grad_mean', torch.tensor(1.0))
        self.register_buffer('update_count', torch.tensor(0))
        self.momentum = 0.99  # EMA momentum

    def forward(self, pred, target, weight=None, positions=None):

        if positions is None:
            raise ValueError("positions required")

        pred_vis = pred[self.key]
        target_grad_mag = torch.clamp(target['grad_mags_transformed'], min=self.threshold)

        # Compute predicted gradient magnitudes
        pred_grad_mag = predict_gradient_magnitudes(pred_vis, positions)
        pred_grad_mag = torch.clamp(pred_grad_mag, min=self.threshold)

        if len(pred_grad_mag) == 0:
            return torch.tensor(0.0, device=pred_vis.device, requires_grad=True)

        # Relative loss: (pred - target) / target
        # This makes it scale-invariant
        relative_diff = (pred_grad_mag - target_grad_mag) / (target_grad_mag + 1e-10)

        loss = F.smooth_l1_loss(
            relative_diff,
            torch.zeros_like(relative_diff),  # Target is 0 (perfect match)
            reduction=self.reduce_method
        )

        return self.weight * loss