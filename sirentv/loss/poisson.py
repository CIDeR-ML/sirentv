import torch
import torch.nn as nn
from sirentv.loss.builder import LOSSES
@LOSSES.register_module()
class WeightedPoissonNLLLoss(nn.Module):
    """
    weighted Poisson negative log-likelihood on linear-domain intensities

    assumes targets are Monte Carlo expectation counts/intensities in linear domain
    and predictions are nonnegative intensities. ignores constant log(k!) term by default.

    inv_xform/n_photon: pred[key]/target[key] are often in a compressed domain (e.g. the
    log-like xform_vis applied to visibility before it ever reaches a loss) rather than raw
    linear-domain intensities, and are usually a normalized fraction (vis/n_photon) rather than
    counts -- Poisson's variance=mean only holds in actual count units. When both are given,
    forward() inverts the domain transform and rescales by n_photon before computing the NLL:
    n_photon * (p - t*log(p)) is gradient-equivalent to the true count-domain NLL on
    (n_photon*p, n_photon*t), up to a p-independent additive constant (see derivation in
    train_sirentv_81_dualpca_poisson.yaml). Leave both None to use pred[key]/target[key] as-is.

    The reported loss also subtracts the p-INDEPENDENT floor value the NLL takes at the true
    optimum p=t: n*(t - t*log(t)). This is exactly the (constant-factor-of-2-dropped) Poisson
    deviance, D = n*[(p - t) - t*log(p/t)], which is convex, always >= 0, and zero exactly at
    p=t -- subtracting a p-independent term changes neither the gradient nor the optimum, only
    the reported/logged magnitude, so the loss actually approaches 0 near convergence instead of
    sitting at a large, target-dependent, uninformative constant that swamps other loss terms
    under `reduction: sum`.
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
        # coerce explicitly: PyYAML parses bare-positive-exponent numbers (e.g. 3.0e7, no sign
        # after "e") as strings, not floats -- n_photon commonly comes from exactly such a
        # config value, so guard here too, not just at the call site
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
            # invert the compressed (e.g. log) domain back to a linear-domain fraction, then
            # rescale to count-equivalent units -- see class docstring for the derivation
            pred_val = self.inv_xform(pred_val)
            target_val = self.inv_xform(target_val)
            if self.n_photon is not None:
                scale = self.n_photon

        # ensure positivity and numerical stability
        pred_lin = torch.clamp(pred_val, min=self.eps)
        target_clamped = torch.clamp(target_val, min=self.eps)
        # poisson nll without constant term: lambda - k * log(lambda)
        nll = pred_lin - target_val * torch.log(pred_lin)
        # p-independent floor at p=target_val -- target_val*log(target_clamped) is 0 (not NaN)
        # when target_val==0, since the eps clamp only affects the argument of log(), not the
        # target_val multiplier in front of it (0 * log(eps) = 0, the standard 0*log(0):=0
        # convention)
        floor = target_val - target_val * torch.log(target_clamped)
        loss = scale * (nll - floor)
        if self.full:
            loss = loss + torch.lgamma(target_val * scale + 1.0)
        loss = weight[self.key] * loss
        return self.weight * self.reduce(loss)
