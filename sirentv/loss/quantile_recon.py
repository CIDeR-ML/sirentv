from __future__ import annotations

import torch
import torch.nn as nn
from sirentv.loss.builder import LOSSES


@LOSSES.register_module()
class WeightedReconstructedQuantileLoss(nn.Module):
    """
    Weighted L2 on a quantile-function-domain target, with a per-bin weight proportional to the
    local density of the TARGET's own adjacent-bin spacing (f(Q(u)) ~ 1/spacing), raised to
    `power`. This is a purely physical-importance weight -- fast-rising, information-dense
    regions of the waveform matter more -- not a confidence weight: there is deliberately no N
    (photon count) or u(1-u) factor here, since photon-count confidence is handled entirely by
    the separate visibility (Poisson NLL) loss.

    If pca_mean/pca_components are given, pred[key]/target[key] are treated as PCA coefficients
    and reconstructed into quantile-function space (mean + coeffs @ components, in whatever
    domain the basis itself was fit in -- e.g. log-quantile -- no extra inverse transform)
    before the weight and loss are computed. This is what makes the same loss usable for both:
      - PCA-coefficient supervision: key="coeffs", pca_mean/pca_components given.
      - Direct quantile-function supervision: key="quantiles", no basis -- pred/target are
        already in quantile-function space, so this reduces to plain density-weighted L2.

    The density weight is detached and computed from the target only (a fixed importance
    multiplier reflecting the true physical signal, not the model's current prediction), and
    normalized to a mean of 1.0 per sample so the overall loss magnitude stays comparable to an
    unweighted baseline.
    """

    def __init__(
        self,
        key: str,
        mask_key: str = None,
        weight: float = 1.0,
        reduce_method: str = "mean",
        power: float = 1.0,
        eps: float = 1e-3,
        pca_mean=None,
        pca_components=None,
        coeff_mean=None,
        coeff_std=None,
    ):
        super().__init__()
        assert reduce_method in ["mean", "sum", "none"], f"Invalid reduction method: {reduce_method}"
        self.reduce = getattr(torch, reduce_method) if reduce_method != "none" else lambda x: x
        self.key = key
        self.mask_key = mask_key
        self.weight = weight
        self.power = float(power)
        self.eps = float(eps)
        self.pca_mean = pca_mean
        self.pca_components = pca_components
        # only needed/used when normalize_coeffs is on for this run -- pred[key]/target[key]
        # are then NORMALIZED coefficients ((raw - mean)/std), which must be denormalized back
        # to raw scale before reconstruct_aligned_raw's mean + coeffs @ components is valid
        # (that formula assumes raw-scale coefficients; reconstructing straight from normalized
        # ones would silently produce meaningless quantile-time values, not an error)
        self.coeff_mean = coeff_mean
        self.coeff_std = coeff_std

    def _reconstruct(self, val, device):
        if self.pca_mean is None:
            return val
        if self.coeff_mean is not None:
            val = val * self.coeff_std.to(device) + self.coeff_mean.to(device)
        mean = self.pca_mean.to(device)
        comp = self.pca_components.to(device)
        return val @ comp + mean

    def _density_weight(self, target_time):
        """(..., n_bins) target values -> (..., n_bins) weight, mean 1 along the last axis."""
        spacing = (target_time[..., 1:] - target_time[..., :-1]).abs().clamp(min=self.eps)
        spacing_at_bin = torch.empty_like(target_time)
        spacing_at_bin[..., 0] = spacing[..., 0]
        spacing_at_bin[..., -1] = spacing[..., -1]
        spacing_at_bin[..., 1:-1] = 0.5 * (spacing[..., :-1] + spacing[..., 1:])
        density = 1.0 / spacing_at_bin
        bin_weight = density ** self.power
        return bin_weight / bin_weight.mean(dim=-1, keepdim=True)

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        weight: dict[str, torch.Tensor] | None = None,
    ):
        device = pred[self.key].device
        pred_time = self._reconstruct(pred[self.key], device)
        target_time = self._reconstruct(target[self.key].to(device), device)

        with torch.no_grad():
            bin_weight = self._density_weight(target_time)

        if weight is None:
            sample_weight = torch.ones_like(pred_time)
        else:
            w = weight.get(self.key, 1.0) if isinstance(weight, dict) else weight
            if isinstance(w, torch.Tensor) and self.pca_mean is not None and w.shape[-1] != pred_time.shape[-1]:
                # a per-sample weight for `key` (e.g. coeffs_weight, from normalize_coeffs /
                # coeff_weight_power) is defined in the PRE-reconstruction, per-component space
                # -- it can't be validly applied to the RECONSTRUCTED per-bin quantity, so
                # ignore it here rather than silently broadcasting into the wrong thing (or
                # crashing). Component-space reweighting and reconstruction-space density
                # weighting are two different axes that were never meant to combine -- this
                # loss is specifically the "replace coefficient-space weighting with bin-space
                # density weighting" design, not an addition to it.
                sample_weight = torch.ones_like(pred_time)
            else:
                sample_weight = w.to(device) if isinstance(w, torch.Tensor) else torch.full_like(pred_time, w)

        full_weight = bin_weight * sample_weight

        if self.mask_key is not None:
            mask = target[self.mask_key]
            while len(mask.shape) < len(target_time.shape):
                mask = mask.unsqueeze(-1)
            true_mask = mask.expand_as(target_time)
            full_weight = full_weight[true_mask]
            pred_time = pred_time[true_mask]
            target_time = target_time[true_mask]

        loss = full_weight * (pred_time - target_time) ** 2
        return self.weight * self.reduce(loss)
