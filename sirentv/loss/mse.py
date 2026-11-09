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
