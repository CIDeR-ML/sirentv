import os

import torch

from sirentv.loss.builder import build_loss as _build_loss_fn


def unwrap_net(net):
    """Unwrap torch.compile (outer) and DDP (inner) wrappers to get the underlying model."""
    from torch.nn.parallel import DistributedDataParallel
    if hasattr(net, '_orig_mod'):
        net = net._orig_mod
    if isinstance(net, DistributedDataParallel):
        net = net.module
    return net


def _clip_grad_(grad_clip_max_norm, net):
    """Clip by global norm (float), or per named subnetwork (dict, e.g. {"v_net": 1.0, ...}).

    Dict keys match a parameter when they equal one segment of its dotted name, so the
    compile/DDP prefixes don't matter. Each group is clipped by its own combined norm, so one
    branch can't throttle another. Every parameter must match exactly one key.
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
    """Build train.loss, injecting the dataset-side constants some losses need."""
    loss_cfg = cfg.get("train", dict()).get("loss", [])
    losses = []
    for c in loss_cfg:
        c = dict(c)
        plib_cfg = cfg.get("compressed_plib", cfg.get("quantile_plib", cfg.get("photonlib", {})))
        if c.get("type") == "WeightedPoissonNLLLoss" and c.get("key") == "v":
            # same xform_vis / n_photon as the dataset, so they can't drift apart
            from slar.transform import partial_xform_vis
            _, inv_xform = partial_xform_vis(cfg.get("transform_vis", {}))
            c.setdefault("inv_xform", inv_xform)
            c.setdefault("n_photon", float(plib_cfg["n_photon"]))  # PyYAML reads 3.0e7 as str
        if c.get("type") == "WeightedReconstructedQuantileLoss" and c.get("key") == "coeffs":
            # coefficient supervision needs the PCA basis to reconstruct quantile space
            if "compressed_plib" in cfg:
                from sirentv.data.compressed import CompressedPLib
                _cplib = CompressedPLib.load(
                    plib_cfg["filepath"], lazy=True, n_components=plib_cfg.get("n_components"),
                )
                c.setdefault("pca_mean", _cplib.pca_mean)
                c.setdefault("pca_components", _cplib.pca_components)
                if bool(plib_cfg.get("normalize_coeffs", False)):
                    # repeats the dataset's one-time stats pass (accepted startup cost)
                    _cplib.compute_coeff_stats()
                    c.setdefault("coeff_mean", _cplib.coeff_mean)
                    c.setdefault("coeff_std", _cplib.coeff_std)
        losses.append(_build_loss_fn(c))
    return losses


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
                continue  # auxiliary weight vector, not a target
            if k == "coeffs" and "coeffs_weight" in target:
                # per-component weight from normalize_coeffs; takes precedence over vis weighting
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


CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config")


def resolve_config(path, archive=False):
    """--config as a path, or a bare name looked up in <repo>/config (config/archive with archive=True)."""
    if archive:
        return os.path.join(CONFIG_DIR, "archive", os.path.basename(path))
    if os.path.exists(path):
        return path
    return os.path.join(CONFIG_DIR, path)
