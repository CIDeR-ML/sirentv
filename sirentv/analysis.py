"""Scalar pred-vs-target metrics, referenced by name from logger.analysis (`func: bias`)."""
from __future__ import annotations

import torch


def _masked_pair(target, pred, key, threshold):
    """(pred, target) for `key`, restricted to target > threshold (None = no mask)."""
    assert key in target, f"key {key} not in data keys {target.keys()}"
    target = target[key]
    pred = pred[key]
    if target.shape != pred.shape:
        raise ValueError(
            f"target and pred must have the same shape {(*target.shape,)} != {(*pred.shape,)}"
        )
    target = target.to(pred.device)
    if threshold is None:
        return pred, target
    mask = target > threshold
    return pred[mask], target[mask]


def bias(
    target: dict[str, torch.Tensor],
    pred: dict[str, torch.Tensor],
    key: str,
    threshold: float = 0.0,
    signed=False,
):
    """Mean of 2|p - t| / (p + t) (signed: 2(p - t) / (p + t)) over target > threshold."""
    p, t = _masked_pair(target, pred, key, threshold)
    if not signed:
        return (2 * torch.abs(p - t) / (p + t)).mean()
    return (2 * (p - t) / (p + t)).mean()


def abs_bias(
    target: dict[str, torch.Tensor],
    pred: dict[str, torch.Tensor],
    key: str,
    signed=False,
    threshold: float = 0.0,
):
    """Mean of |t - p| (signed: t - p) over target > threshold."""
    p, t = _masked_pair(target, pred, key, threshold)
    return torch.abs(t - p).mean() if not signed else (t - p).mean()
