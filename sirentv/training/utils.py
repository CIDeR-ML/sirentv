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


def backward_step(loss, opt, amp, scaler, grad_clip_max_norm, net):
    """AMP-aware backward pass, gradient clipping, and optimizer step."""
    if amp:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        if grad_clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_max_norm)
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        if grad_clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_max_norm)
        opt.step()


def step_scheduler(scheduler, metric):
    """Advance a scheduler using the argument contract expected by PyTorch."""
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metric)
    else:
        scheduler.step()


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
            if k in weight_cfg and weight_cfg[k].get("enable", False):
                weights[k] = get_weight_by_vis(
                    target[k],
                    factor=weight_cfg[k].get("factor", None),
                    threshold=weight_cfg[k].get("threshold", 1e-8),
                )
            else:
                weights[k] = 1.0
        return weights

    return weight_fn
