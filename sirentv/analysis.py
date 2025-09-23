
from __future__ import annotations
from typing import List

import torch.nn.functional as F
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import wandb
import torch
from typing import Literal
from sirentv.utils.transform import pdf_to_cdf, cdf_to_pdf

def _products_from_pdf(pdf):
    v = pdf.sum(-1)
    t_pdf = F.normalize(pdf, p=1, dim=-1)
    t_cdf = t_pdf.cumsum(-1)
    return dict(
        t_pdf=t_pdf,
        t_cdf=t_cdf,
        v=v,
    )

@torch.no_grad()
def get_pred_target(dataloader, net, max_voxels=pow(2, 18)): # 262144 vox max
    n_voxels = min(len(dataloader._plib), max_voxels)
    vox_ids = torch.randperm(len(dataloader._plib), device=dataloader._plib.device)[:n_voxels]
    positions = dataloader._plib.meta.voxel_to_coord(vox_ids)

    batch_size = 2048
    pred_t = []
    curr_idx = 0
    for i in range(len(positions) // batch_size):
        curr_idx = i * batch_size
        pred_t_ = net.visibility(positions[curr_idx : curr_idx + batch_size]) # (B, N_pmt, N_tick)
        pred_t.append(pred_t_.cpu())
    pred_t_pdf_unnorm = torch.cat(pred_t, dim=0) # (B, N_pmt, N_tick)
    target_t_pdf_unnorm = dataloader._plib[vox_ids].squeeze().cpu() #

    pred = _products_from_pdf(pred_t_pdf_unnorm)
    target = _products_from_pdf(target_t_pdf_unnorm)
    return pred, target


def log_pred_target(pred, target, name="pred_vs_target"):
    # get bounds
    nonzero_pred = pred[pred > 0]
    nonzero_target = target[target > 0]

    xmin = ymin = max(min(nonzero_pred.min().item(), nonzero_target.min().item()), 1e-8)
    xmax = ymax = min(max(nonzero_pred.max().item(), nonzero_target.max().item()), 1e0)

    h = (
        Hist.new.Log(250, xmin, xmax, name="x", label="Predicted")
        .Log(250, ymin, ymax, name="y", label="Target")
        .Int64()
    )
    batch_size = len(pred) // 2048
    for i in range(batch_size):
        h.fill(
            pred[i * 2048 : (i + 1) * 2048].ravel(),
            target[i * 2048 : (i + 1) * 2048].ravel(),
        )

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    mesh = ax.pcolormesh(
        h.axes[0].edges, h.axes[1].edges, h.values().T, norm=LogNorm(), cmap="viridis"
    )
    print(xmin, xmax, ymin, ymax)
    ax.plot([xmin, xmax], [ymin, ymax], color="red")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    ax.set_title(name)
    plt.colorbar(mesh, label="Counts")
    
    # Log the plot to wandb
    wandb.log({name: wandb.Image(fig)})
    
    plt.close(fig)  # Close the figure to free up memory
    del h

def log_imshow(tensor, name="imshow"):
    fig, ax = plt.subplots()
    img = ax.imshow(tensor, cmap="viridis", aspect="auto", norm=LogNorm(), origin="lower")
    fig.colorbar(img, ax=ax)
    ax.set_title(name)
    wandb.log({name: wandb.Image(fig)})
    plt.close(fig)

def log_line(tensor, name="line"):
    fig, ax = plt.subplots()
    ax.plot(tensor)
    ax.set_title(name)
    wandb.log({name: wandb.Image(fig)})
    plt.close(fig)


def log_hist(tensor, name="hist"):
    fig, ax = plt.subplots()
    ax.hist(tensor)
    ax.set_title(name)
    wandb.log({name: wandb.Image(fig)})
    plt.close(fig)

def bias(
    target: dict[str, torch.Tensor],
    pred: dict[str, torch.Tensor],
    key: str,
    threshold: float = 0.0,
    signed=False,
):
    """
    Function to compute the visibility bias (the mean of 2 * |target - pred| / (target + pred))

    Parameters
    ----------
    target : torch.Tensor
        The reference visibility based on which the bias is calculated.
    pred : torch.Tensor
        The subject visibility for which the bias is calculated.
    threshold : float
        The visibility lowest threshold. The visibility bias is computed only
        for the instances for which the reference (target) tensor contains the
        visibility value above this threshold.

    Returns
    -------
    torch.Tensor
        The model visibility bias.
    """
    assert key in target, f"key {key} not in data keys {target.keys()}"
    target = target[key]
    pred = pred[key]

    if target.shape != pred.shape:
        raise ValueError(
            f"target and pred must have the same shape {(*target.shape,)} != {(*pred.shape,)}"
        )

    mask = target > threshold
    p = pred[mask]
    t = target[mask]

    if not signed:
        bias = (2 * torch.abs(p - t) / (p + t)).mean()
    else:
        bias = (2 * (p - t) / (p + t)).mean()
    return bias

def abs_bias(
    target: dict[str, torch.Tensor],
    pred: dict[str, torch.Tensor],
    key: str,
    signed=False,
):
    """
    Function to compute the absolute bias (the mean of |target - pred|)

    Parameters
    ----------
    target : torch.Tensor
        Some reference target based on which the bias is calculated.
    pred : torch.Tensor
        Prediction for target on which which the bias is calculated.

    Returns
    -------
    torch.Tensor
        The model absolute bias.

    """
    assert key in target, f"key {key} not in data keys {target.keys()}"
    target = target[key]
    pred = pred[key]

    if target.shape != pred.shape:
        raise ValueError(
            f"target and pred must have the same shape {(*target.shape,)} != {(*pred.shape,)}"
        )
    return torch.abs(target - pred).mean() if not signed else (target - pred).mean()
