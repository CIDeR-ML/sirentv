import torch
import torch.nn as nn
from sirentv.loss.builder import LOSSES


@LOSSES.register_module()
class WeightedPoissonNLLLoss(nn.Module):
    """Weighted Poisson NLL, lambda - k*log(lambda), on linear-domain intensities.

    inv_xform / n_photon: if given, pred/target are first mapped back from the xform_vis domain
    and the NLL is scaled by n_photon, which makes the fraction-domain loss gradient-equivalent
    to the count-domain NLL (Poisson variance == mean only holds in counts).
    The p-independent value at p == t is subtracted so the logged loss tends to 0 at
    convergence; gradients and optimum are unchanged. full=True adds the log(k!) term.
    """

    def __init__(self, key: str, weight=1.0, reduce_method=torch.mean, full: bool = False, eps: float = 1e-8,
                 inv_xform=None, n_photon: float = None):
        super().__init__()
        self.reduce = reduce_method
        self.full = bool(full)
        self.eps = float(eps)
        self.key = key
        self.weight = weight
        self.inv_xform = inv_xform
        # PyYAML reads e.g. 3.0e7 (no exponent sign) as a string
        self.n_photon = float(n_photon) if n_photon is not None else None

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
        pred_val = pred[self.key]
        target_val = target[self.key].to(device)

        scale = 1.0
        if self.inv_xform is not None:
            pred_val = self.inv_xform(pred_val)
            target_val = self.inv_xform(target_val)
            if self.n_photon is not None:
                scale = self.n_photon

        pred_lin = torch.clamp(pred_val, min=self.eps)
        target_clamped = torch.clamp(target_val, min=self.eps)
        nll = pred_lin - target_val * torch.log(pred_lin)
        # clamped log keeps the floor 0 (not NaN) where target_val == 0
        floor = target_val - target_val * torch.log(target_clamped)
        loss = scale * (nll - floor)
        if self.full:
            loss = loss + torch.lgamma(target_val * scale + 1.0)
        loss = weight[self.key] * loss
        return self.weight * self.reduce(loss)
