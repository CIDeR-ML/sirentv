import torch
import torch.nn as nn

class WeightedPoissonNLLLoss(nn.Module):
    """
    weighted Poisson negative log-likelihood on linear-domain intensities

    assumes targets are Monte Carlo expectation counts/intensities in linear domain
    and predictions are nonnegative intensities. ignores constant log(k!) term by default.
    """

    def __init__(self, reduce_method=torch.mean, full: bool = False, eps: float = 1e-8):
        super().__init__()
        self.reduce = reduce_method
        self.full = bool(full)
        self.eps = float(eps)
        # signal that this loss expects linear-domain inputs
        self.requires_linear_domain = True

    def forward(self, pred, target, weight=1.0):
        # ensure positivity and numerical stability
        pred_lin = torch.clamp(pred, min=self.eps)
        # poisson nll without constant term: lambda - k * log(lambda)
        loss = pred_lin - target * torch.log(pred_lin)
        if self.full:
            loss = loss + torch.lgamma(target + 1.0)
        if weight is not None:
            loss = weight * loss
        return self.reduce(loss)
