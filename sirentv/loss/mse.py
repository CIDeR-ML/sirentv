from __future__ import annotations

import torch
import torch.nn as nn
from sirentv.loss.builder import LOSSES


@LOSSES.register_module()
class WeightedL2Loss(nn.Module):
    def __init__(self, key: str, mask_key: str = None, weight=1.0, reduce_method="mean"):
        super().__init__()
        assert reduce_method in ["mean", "sum", "none"], f"Invalid reduction method: {reduce_method}"
        self.reduce = getattr(torch, reduce_method) if reduce_method != "none" else lambda x: x
        self.key = key
        self.mask_key = mask_key
        self.weight = weight

    def forward(
            self,
            pred: dict[str, torch.Tensor],
            target: dict[str, torch.Tensor],
            weight: dict[str, torch.Tensor] | None = None,
        ):
        if weight is None:
            weight = {self.key: torch.ones_like(pred[self.key])}
        device = pred[self.key].device
        w = weight.get(self.key, 1.0) if isinstance(weight, dict) else weight
        weight_masked = w.to(device) if isinstance(w, torch.Tensor) else w
        target_masked = target[self.key].to(device)
        pred_masked = pred[self.key]
        if self.mask_key is not None:
            mask = target[self.mask_key]
            while len(mask.shape) < len(target_masked.shape):
                mask = mask.unsqueeze(-1)
            true_mask = mask.expand_as(target_masked)
            if isinstance(weight_masked, torch.Tensor):
                weight_masked = weight_masked[true_mask]
            target_masked = target_masked[true_mask]
            pred_masked = pred_masked[true_mask]

        loss = weight_masked * (pred_masked - target_masked) ** 2
        return self.weight * self.reduce(loss)

@LOSSES.register_module()
class L2Loss(WeightedL2Loss):
    def __init__(self, key: str, mask_key: str = None, weight=1.0, reduce_method="mean"):
        super().__init__(key, mask_key, weight, reduce_method)
    def forward(
            self,
            pred: dict[str, torch.Tensor],
            target: dict[str, torch.Tensor],
            weight: dict[str, torch.Tesnor] | None = None
        ):
        return super().forward(pred, target, weight=None)


#TODO: add key/weight
@LOSSES.register_module()
class UncertainMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weight, log_sigma):
        sigma = torch.exp(log_sigma)
        loss = torch.mean(weight * (((target - pred) ** 2) / (sigma**2) + log_sigma))
        return loss


@LOSSES.register_module()
class VisibilityGradient_L2Loss(WeightedL2Loss):
    """Gradient magnitude loss using any base loss function"""

    def __init__(self, key: str = 'grad_mags_transformed', mask_key: str = None, weight=1.0, reduce_method="mean", threshold=1.E-9):
        super().__init__(key, mask_key, weight, reduce_method)
        self.threshold = threshold
        self.key = key
    def forward(self, pred, target, weight=None):

        target_grad_mag = torch.clamp(target[self.key], min=self.threshold)
        pred_grad_mag = torch.clamp(pred[self.key], min=self.threshold)

        if len(pred_grad_mag) == 0:
            return torch.tensor(0.0, device=pred['v'].device, requires_grad=True)

        if len(pred_grad_mag) == 0:
            return torch.tensor(0.0, device=pred['v'].device, requires_grad=True)

        # Relative loss: (pred - target) / target
        relative_diff = (pred_grad_mag - target_grad_mag) / (target_grad_mag + 1e-10)

        pred[self.key] = relative_diff
        target[self.key] = torch.zeros_like(relative_diff)

        return super().forward(pred, target, weight=None)