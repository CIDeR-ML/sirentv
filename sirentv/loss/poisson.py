import torch
import torch.nn as nn
from sirentv.loss.builder import LOSSES
@LOSSES.register_module()
class WeightedPoissonNLLLoss(nn.Module):
    """
    weighted Poisson negative log-likelihood on linear-domain intensities

    assumes targets are Monte Carlo expectation counts/intensities in linear domain
    and predictions are nonnegative intensities. ignores constant log(k!) term by default.
    """

    def __init__(self, key: str, weight=1.0, reduce_method=torch.mean, full: bool = False, eps: float = 1e-8):
        super().__init__()
        self.reduce = reduce_method
        self.full = bool(full)
        self.eps = float(eps)
        self.key = key
        self.weight = weight

    def forward(
            self,
            pred: dict[str, torch.Tensor],
            target: dict[str, torch.Tensor],
            weight: dict[str, torch.Tensor] | None,
        ):
        if weight is None:
            weight = {self.key: torch.ones_like(pred[self.key])}
        device = pred[self.key].device
        weight[self.key] = weight[self.key].to(device)
        target[self.key] = target[self.key].to(device)
        # ensure positivity and numerical stability
        pred_lin = torch.clamp(pred[self.key], min=self.eps)
        # poisson nll without constant term: lambda - k * log(lambda)
        loss = pred_lin - target[self.key] * torch.log(pred_lin)
        if self.full:
            loss = loss + torch.lgamma(target[self.key] + 1.0)
        loss = weight[self.key] * loss
        return self.weight * self.reduce(loss)
