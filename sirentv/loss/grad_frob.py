from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from sirentv.loss.builder import LOSSES


@LOSSES.register_module()
class WeightedGradFrobLoss(nn.Module):
    """
    Matches the model's predicted per-(voxel, PMT) SQUARED spatial-gradient Frobenius norm
    (pred[f"{key}_grad_frob"], from compute_grad_frob_hutchinson[_aggregate] -- create_graph=True,
    a genuine trainable signal) against the true finite-difference one (target[f"{key}_grad_frob"],
    from compute_grad_frob_target[_exact_projected]) via smooth-L1 on the ABSOLUTE difference.
    Both sides are the SQUARED Frobenius norm (no sqrt) specifically because the Hutchinson
    identity E[||grad(proj)||^2] = ||J||_F^2 is exact/unbiased at ANY number of projections,
    including the prediction side's single projection per step -- taking a sqrt afterwards
    would reintroduce Jensen's-inequality bias (E[sqrt(X)] < sqrt(E[X]) for concave sqrt),
    which is systematically worse the fewer projections are averaged, i.e. worst exactly on the
    prediction side.

    Deliberately not a relative/ratio formula on the COMPARED VALUES themselves (e.g.
    2*(pred-target)/(pred+target+eps) in place of the smooth-L1 argument): this session
    repeatedly found that style of relative loss becomes unstable wherever the compared
    quantities are near zero (see the earlier VisibilityGradient_SmoothL1Loss discussion, which
    has that exact issue on top of being create_graph=False and non-functional as a training
    signal to begin with -- this loss fixes both problems rather than inheriting either).
    dynamic_weight (below) is a DIFFERENT thing from that rejected approach: it's a multiplier
    on the smooth-L1 RESIDUAL, not a replacement for it, so pred=target is still the unique
    zero of the loss regardless of the weight's value.

    If pred[f"{key}_grad_frob"] isn't present at all, contributes zero rather than raising --
    this happens whenever train.py's per-key grad_supervision_keys warmup hasn't reached this
    key's start epoch yet (see train.py), so the model wasn't asked to compute it this step.

    scale: multiplies pred_val/target_val (both sides, so what's being matched is unchanged)
    before the smooth-L1 comparison. Exists because grad_frob's natural magnitude can be tiny
    for reasons unrelated to how well pred matches target -- e.g. for key="v", grad_frob is the
    spatial derivative of xform_vis(vis/n_photon), and dividing by n_photon (~1e7-1e8) crushes
    the scale down to ~1e-8 regardless of fit quality, making weight=1.0 numerically negligible
    next to other loss terms under `reduction: sum`. Unlike WeightedPoissonNLLLoss's n_photon
    rescale, there's no exact algebraic identity here (xform_vis may be nonlinear), so this is
    an empirical knob, not a derived constant -- start from n_photon for key="v" (undoes exactly
    the division that crushed the scale) and check logged magnitudes.

    dynamic_weight: if True, each entry's smooth-L1 term is multiplied by
    1 / (0.5*(pred_val + target_val) + dynamic_weight_eps) BEFORE reduction. Two effects,
    both wanted:
      1. Counteracts the extra concentration from comparing in the SQUARED domain (forced by
         the unbiasedness argument above): the true field's dynamic range is far more extreme
         once squared than in the original magnitude domain (e.g. a 100x magnitude spread
         becomes a 10000x squared spread), so an unweighted loss here is effectively MORE
         concentrated on high-gradient (near-PMT) voxels than a direct gradient-VECTOR loss
         (like the original SIREN paper's) would be for the same physical field. Dividing by
         (roughly) the local scale brings the per-entry weighting back down from
         ~linear-in-squared-magnitude to ~linear-in-magnitude.
      2. Self-damping against exploding gradients: unlike scale/weight (fixed constants), this
         denominator uses pred_val too, and is DELIBERATELY NOT DETACHED (contrast every other
         weight in this codebase -- e.g. compute_grad_frob_hutchinson_aggregate's vis_weight,
         WeightedReconstructedQuantileLoss's bin_weight -- which are detached on purpose so the
         model can't move the weight instead of the residual). Here that's intentional: for
         pred_val >> target_val, smooth-L1 is linear (~pred_val) and the weight is ~1/pred_val,
         so their product approaches a CONSTANT rather than growing with pred_val -- an already-
         diverged outlier's own gradient contribution saturates instead of exploding further.
         The global minimum is unaffected (at pred_val=target_val the smooth-L1 term is exactly
         0, so the product is 0 regardless of the weight), so this can't create a false minimum.
    dynamic_weight_eps: floor on the denominator above -- without it, voxels where BOTH
    pred_val and target_val are near zero (expected in the far field) would divide by
    something close to 0 and blow the weight up, amplifying whatever prediction-side M=1 noise
    floor exists there instead of damping it -- the same near-zero-relative-quantity failure
    mode this docstring's second paragraph already flags for a plain ratio loss, now guarded
    against here too.
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
            # pred_val intentionally NOT detached -- see dynamic_weight's docstring above for
            # why that's deliberate here, unlike every other weight in this codebase.
            dyn_w = 1.0 / (0.5 * (pred_val + target_val) + self.dynamic_weight_eps)
            loss = dyn_w * loss
        return self.weight * self.reduce(loss)
