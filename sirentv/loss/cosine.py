import torch
import torch.nn as nn
import torch.nn.functional as F
from sirentv.loss.builder import LOSSES


@LOSSES.register_module()
class WeightedCosineDissimilarity(nn.Module):
    """
    A simple loss module that implements a weighted cosine dissimilarity loss

    To be used when training on the *shape* of an output (e.g., PMT waveform), and not
    the explicit values.
    """

    def __init__(self, key: str, weight=1.0, reduce_method="mean"):
        super().__init__()
        assert reduce_method in ["mean", "sum", "none"], f"Invalid reduction method: {reduce_method}"
        self.reduce = getattr(torch, reduce_method) if reduce_method != "none" else lambda x: x
        self.key = key
        self.weight = weight

    def forward(self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor], weight: torch.Tensor):
        pred = pred[self.key]
        target = target[self.key]
        device = pred.device
        target = target.to(device)

        pred_norm = F.normalize(pred, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)
        # normalize_weights = F.normalize(weight, p=2, dim=-1) if hasattr(weight, 'norm') else weight
        loss = (weight * (1 - (pred_norm * target_norm))).mean(dim=-1)
        return self.weight * self.reduce(loss)
    