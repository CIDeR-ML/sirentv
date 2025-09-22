from hist import Hist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import wandb
import torch

def get_pred_target(dataloader, net):
    cache = dataloader._cache
    n_voxels = len(dataloader._plib)
    vox_ids = torch.arange(n_voxels, device=dataloader._plib.device)
    positions = dataloader._plib.meta.voxel_to_coord(vox_ids)

    batch_size = 2048
    pred = []
    curr_idx = 0
    for i in range(len(positions) // batch_size):
        curr_idx = i * batch_size
        out_vis = net.visibility(positions[curr_idx : curr_idx + batch_size])
        pred.append(out_vis.detach().cpu())
    pred = torch.cat(pred).unsqueeze(-1)
    target = cache["value"].squeeze().detach().cpu()
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
