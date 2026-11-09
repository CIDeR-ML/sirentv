from __future__ import annotations

import torch
import torch.nn as nn
from sirentv.loss.builder import LOSSES


@LOSSES.register_module()
class WeightedReconstructedQuantileLoss(nn.Module):
    """L2 in quantile-function space, each bin weighted by the target's local density (1/spacing)**power.

    The weight is physical importance (fast-rising parts of the waveform), not photon-count
    confidence, which the visibility loss handles. It is detached, computed from the target
    only, and normalized to mean 1 per sample.
    With pca_mean/pca_components, pred[key]/target[key] are PCA coefficients and are first
    reconstructed (mean + coeffs @ components, in the basis' own domain); coeff_mean/coeff_std
    undo normalize_coeffs first. Without a basis, key="quantiles" is used directly.
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
                # a per-component coefficient weight has no meaning per reconstructed bin
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
