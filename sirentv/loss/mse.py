from __future__ import annotations

import torch
import torch.nn as nn
from sirentv.loss.builder import LOSSES

@LOSSES.register_module()
class WeightedL2Loss(nn.Module):
    def __init__(self, key: str, weight=1.0, reduce_method="mean"):
        super().__init__()
        assert reduce_method in ["mean", "sum", "none"], f"Invalid reduction method: {reduce_method}"
        self.reduce = getattr(torch, reduce_method) if reduce_method != "none" else lambda x: x
        self.key = key
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
        weight[self.key] = weight[self.key].to(device)
        target[self.key] = target[self.key].to(device)

        loss = weight[self.key] * (pred[self.key] - target[self.key]) ** 2
        return self.weight * self.reduce(loss)

@LOSSES.register_module()
class L2Loss(WeightedL2Loss):
    def __init__(self, key: str, weight=1.0, reduce_method="mean"):
        super().__init__(key, weight, reduce_method)
    def forward(
            self,
            pred: dict[str, torch.Tensor],
            target: dict[str, torch.Tensor],
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
