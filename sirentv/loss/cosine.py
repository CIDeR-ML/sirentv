import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCosineDissimilarity(nn.Module):
    """
    A simple loss module that implements a weighted cosine dissimilarity loss

    To be used when training on the *shape* of an output (e.g., PMT waveform), and not
    the explicit values.
    """

    def __init__(self, reduce_method=torch.mean):
        super().__init__()
        self.reduce = reduce_method
    
    def forward(self, pred, target, weight=1.0):
        pred_norm = F.normalize(pred, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)
        # normalize_weights = F.normalize(weight, p=2, dim=-1) if hasattr(weight, 'norm') else weight
        loss = (weight * (1 - (pred_norm * target_norm))).mean(dim=-1)
        return self.reduce(loss)
    