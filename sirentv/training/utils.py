import torch
import torch.nn as nn

from sirentv.loss.builder import build_loss as _build_loss_fn
from sirentv.loss.builder import build_regularizer as _build_regularizer_fn


def unwrap_net(net):
    """Unwrap torch.compile (outer) and DDP (inner) wrappers to get the underlying model."""
    from torch.nn.parallel import DistributedDataParallel
    if hasattr(net, '_orig_mod'):
        net = net._orig_mod
    if isinstance(net, DistributedDataParallel):
        net = net.module
    return net


def _clip_grad_(grad_clip_max_norm, net):
    """Applies grad_clip_max_norm, either a single float (one global clip across every
    parameter -- the original behavior) or a dict (per-NAMED-SUBNETWORK clip, e.g.
    {"vt0_net": 1.0, "coeff_net": 1.0} for DualPcaSiren -- each key's matching parameters are
    clipped TOGETHER as one group, by their combined norm, so one branch's gradient can no
    longer dominate the global norm and throttle an unrelated branch's otherwise-fine gradient
    via a single shared clip.

    Matching is by dotted-path SEGMENT, e.g.
    "model.vt0_net.net.0.linear.weight"), optionally wrapped again in DDP's "module." prefix --
    neither wrapper's prefix, nor the surrounding path depth, should matter for matching a bare
    submodule name like "vt0_net". A parameter matching more than one key, or none, is an error
    """
    if grad_clip_max_norm is None:
        return
    if isinstance(grad_clip_max_norm, dict):
        groups = {key: [] for key in grad_clip_max_norm}
        unmatched = []
        for name, param in net.named_parameters():
            parts = name.split(".")
            matches = [key for key in grad_clip_max_norm if key in parts]
            if len(matches) > 1:
                raise ValueError(f"Parameter {name!r} matches multiple grad_clip_max_norm keys: {matches}")
            if not matches:
                unmatched.append(name)
                continue
            groups[matches[0]].append(param)
        if unmatched:
            raise ValueError(
                f"{len(unmatched)} parameter(s) matched no grad_clip_max_norm key (e.g. {unmatched[0]!r}) "
                f"-- every parameter must be covered by some key, or it silently goes unclipped."
            )
        for key, params in groups.items():
            if params:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip_max_norm[key])
    elif isinstance(grad_clip_max_norm, float):
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_max_norm)
    else:
        raise ValueError(f"Invalid grad_clip_max_norm type: {type(grad_clip_max_norm)}")


def backward_step(loss, opt, amp, scaler, grad_clip_max_norm, net):
    """AMP-aware backward pass, gradient clipping, and optimizer step."""
    if amp:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        _clip_grad_(grad_clip_max_norm, net)
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        _clip_grad_(grad_clip_max_norm, net)
        opt.step()


def get_weight_by_vis(vis, factor=None, threshold=1e-8):
    if factor is None:
        factor = 1 / torch.max(vis.clamp(min=1e-8))
    w = vis * factor
    w[w < threshold] = 1.0
    return w


def build_losses(cfg):
    loss_cfg = cfg.get("train", dict()).get("loss", [])
    losses = []
    for c in loss_cfg:
        c = dict(c)
        # plib_cfg mirrors the same "compressed_plib, falling back to quantile_plib/photonlib"
        # lookup the datasets themselves use, so pca/dualpca and quantile-family configs both
        # work here without needing separate branches
        plib_cfg = cfg.get("compressed_plib", cfg.get("quantile_plib", cfg.get("photonlib", {})))
        if c.get("type") == "WeightedPoissonNLLLoss" and c.get("key") == "v":
            # reuse the same xform_vis/n_photon the model and dataset already build from, rather
            # than duplicating these values into the loss config entry (that duplication is
            # exactly what caused the stale raw_n_photon drift bug in the eval notebook)
            from slar.transform import partial_xform_vis
            _, inv_xform = partial_xform_vis(cfg.get("transform_vis", {}))
            c.setdefault("inv_xform", inv_xform)
            # float(...) -- PyYAML's float regex requires an explicit sign after "e" (1.0e-3
            # parses as a float, 3.0e7 does not), so bare-positive-exponent values like this
            # codebase's own n_photon: 3.0e7 configs parse as strings; every other n_photon
            # read site already defends against this (e.g. compressed.py's
            # float(plib_cfg.get("n_photon", 1.0))), this one needs the same treatment
            c.setdefault("n_photon", float(plib_cfg["n_photon"]))
        if c.get("type") == "WeightedReconstructedQuantileLoss" and c.get("key") == "coeffs":
            # only the PCA-coefficient-supervision variant needs the basis to reconstruct into
            # quantile-function space -- direct quantile-function supervision (key="quantiles")
            # has no compressed_plib section at all, so pca_mean/pca_components stay None and
            # the loss skips reconstruction (see WeightedReconstructedQuantileLoss docstring)
            if "compressed_plib" in cfg:
                from sirentv.data.compressed import CompressedPLib
                _cplib = CompressedPLib.load(
                    plib_cfg["filepath"], lazy=True, n_components=plib_cfg.get("n_components"),
                )
                c.setdefault("pca_mean", _cplib.pca_mean)
                c.setdefault("pca_components", _cplib.pca_components)
                if bool(plib_cfg.get("normalize_coeffs", False)):
                    # pred[coeffs]/target[coeffs] are normalized ((raw-mean)/std) when this is
                    # on -- reconstruction needs raw-scale coefficients, so the loss denormalizes
                    # internally using these. NOTE: this repeats the same full-dataset
                    # compute_coeff_stats() pass the dataset itself already does at startup (see
                    # CompressedPLibDataset.__init__) -- a known, currently-accepted redundant
                    # one-time cost, not a per-step one; only worth deduplicating if this startup
                    # overhead actually becomes a problem in practice.
                    _cplib.compute_coeff_stats()
                    c.setdefault("coeff_mean", _cplib.coeff_mean)
                    c.setdefault("coeff_std", _cplib.coeff_std)
        losses.append(_build_loss_fn(c))
    return losses


def build_regularizer(cfg) -> nn.Module | None:
    regularizer_cfg = cfg.get("train", dict()).get("regularization", None)
    if regularizer_cfg is None:
        return None
    return _build_regularizer_fn(regularizer_cfg)


def build_logger(cfg, net, rank=0):
    from sirentv.utils.log import CSVLogger, WandbLogger
    logger_type = cfg.get("logger", dict()).get("type", "csv")
    if logger_type == "csv":
        return CSVLogger(cfg, rank=rank)
    return WandbLogger(cfg, rank=rank)


def compute_loss(pred, target, losses, weights):
    losses_out = {}
    for loss in losses:
        curr_loss = loss(pred, target, weights)
        cls_name = loss.__class__.__name__.lower()
        losses_out[f"{cls_name}_{loss.key}"] = curr_loss
    return losses_out


def build_weight_fn(cfg):
    """Build a weight function from config. Returns callable: target_dict -> weights_dict."""
    weight_cfg = cfg.get("data", {}).get("weight", {})

    def weight_fn(target):
        weights = {}
        for k in target.keys():
            if k == "coeffs_weight":
                continue  # auxiliary per-component weight vector, not a prediction target itself
            if k == "coeffs" and "coeffs_weight" in target:
                # per-component magnitude-proportional weight (see CompressedPLib.compute_coeff_stats)
                # -- only present when normalize_coeffs is on, takes precedence over the
                # vis-based weighting below since it's a different (per-component, not
                # per-PMT/visibility) axis of weighting
                weights[k] = target["coeffs_weight"]
            elif k in weight_cfg and weight_cfg[k].get("enable", False):
                weights[k] = get_weight_by_vis(
                    target[k],
                    factor=weight_cfg[k].get("factor", None),
                    threshold=weight_cfg[k].get("threshold", 1e-8),
                )
            else:
                weights[k] = 1.0
        return weights

    return weight_fn
