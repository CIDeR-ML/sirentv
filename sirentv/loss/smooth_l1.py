from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from sirentv.loss.builder import LOSSES


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
        loss = weight[self.key] * F.smooth_l1_loss(pred[self.key] - target[self.key], **self.kwargs)
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
