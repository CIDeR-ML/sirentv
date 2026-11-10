from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from sirentv.loss.builder import LOSSES


@LOSSES.register_module()
class WeightedGradFrobLoss(nn.Module):
    """Smooth-L1 between predicted and target SQUARED spatial-gradient Frobenius norms.

    pred[f"{key}_grad_frob"] comes from compute_grad_frob_hutchinson[_aggregate] (differentiable),
    target[f"{key}_grad_frob"] from compute_grad_frob_target[_exact_projected]. Both stay squared:
    the Hutchinson estimate of ||J||_F^2 is unbiased at any number of projections, a sqrt would
    bias it (Jensen). Absolute, not relative, residual: ratio losses are unstable near zero.
    Returns 0 when the prediction is absent (the key's grad_supervision warmup has not started).

    scale: multiplies both sides before the comparison, to lift tiny natural magnitudes
        (e.g. v after dividing by n_photon) to where `weight` matters. Empirical, not derived.
    dynamic_weight: multiply each residual by 1 / (0.5*(pred + target) + dynamic_weight_eps).
        Undoes the extra high-gradient concentration of the squared domain, and since pred is
        deliberately NOT detached, an outlier's gradient saturates instead of growing.
        The minimum (pred == target) is unchanged. The eps keeps far-field ~0/~0 voxels finite.
    """

    def __init__(self, key: str, weight: float = 1.0, reduce_method: str = "mean", beta: float = 1.0,
                 scale: float = 1.0, dynamic_weight: bool = False, dynamic_weight_eps: float = 1e-3):
        super().__init__()
        assert reduce_method in ["mean", "sum", "none"], f"Invalid reduction method: {reduce_method}"
        self.reduce = getattr(torch, reduce_method) if reduce_method != "none" else lambda x: x
        self.key = key
        self.beta = beta
        self.weight = float(weight)
        self.scale = float(scale)
        self.dynamic_weight = bool(dynamic_weight)
        self.dynamic_weight_eps = float(dynamic_weight_eps)

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        weight: dict[str, torch.Tensor] | None = None,
    ):
        grad_key = f"{self.key}_grad_frob"
        if grad_key not in pred:
            return torch.tensor(0.0, device=pred["v"].device)
        pred_val = pred[grad_key] * self.scale
        target_val = target[grad_key].to(pred_val.device) * self.scale
        loss = F.smooth_l1_loss(pred_val, target_val, beta=self.beta, reduction="none")
        if self.dynamic_weight:
            dyn_w = 1.0 / (0.5 * (pred_val + target_val) + self.dynamic_weight_eps)
            loss = dyn_w * loss
        return self.weight * self.reduce(loss)
