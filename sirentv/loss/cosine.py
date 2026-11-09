from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from sirentv.loss.builder import LOSSES


@LOSSES.register_module()
class WeightedCosineDissimilarity(nn.Module):
    """Weighted cosine dissimilarity: trains the *shape* of an output (e.g. a waveform), not its values."""

    def __init__(self, key: str, weight=1.0, reduce_method="mean"):
        super().__init__()
        assert reduce_method in ["mean", "sum", "none"], f"Invalid reduction method: {reduce_method}"
        self.reduce = getattr(torch, reduce_method) if reduce_method != "none" else lambda x: x
        self.key = key
        self.weight = weight

    def forward(self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor], weight: torch.Tensor):
        pred = pred[self.key]
        target = target[self.key].to(pred.device)
        pred_norm = F.normalize(pred, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)
        loss = (weight * (1 - (pred_norm * target_norm))).mean(dim=-1)
        return self.weight * self.reduce(loss)
