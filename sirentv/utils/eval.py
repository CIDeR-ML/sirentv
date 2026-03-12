"""
Evaluation utilities for compressed PLib: compare pred vs target CDFs, visibility, t0.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Optional, Sequence, Union

from sirentv.data.compressed import CompressedPLib


class Evaluator:
    """Compare predicted (vis, t0, coeffs) vs ground truth; compute metrics and plots."""

    EDGE_BINS = 30

    def __init__(self, plib: CompressedPLib):
        self.plib = plib

    def compare(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        n_bins: int = 1000,
        vis_threshold: float = 100.0,
    ) -> Dict[str, float]:
        """
        Reconstruct CDFs from pred and target; compute RMSE and visibility/t0 errors.
        pred, target: dicts with keys "vis", "t0", "coeffs" (tensors, batched).
        Returns metrics dict.
        """
        pred_vis = pred["vis"]
        pred_t0 = pred["t0"].long().clamp(min=0)
        pred_coeffs = pred["coeffs"]
        true_vis = target["vis"]
        true_t0 = target["t0"].long().clamp(min=0)
        true_coeffs = target["coeffs"]

        pred_cdf = self.plib.reconstruct_cdf(pred_coeffs, pred_t0, n_bins=n_bins)
        true_cdf = self.plib.reconstruct_cdf(true_coeffs, true_t0, n_bins=n_bins)

        bright = true_vis > vis_threshold
        cdf_diff = (pred_cdf - true_cdf) * bright.unsqueeze(-1).float()
        n_bright = bright.sum().item() * n_bins
        if n_bright > 0:
            cdf_rmse_full = torch.sqrt((cdf_diff ** 2).sum() / n_bright).item()
        else:
            cdf_rmse_full = float("nan")

        edge_slice = slice(self.plib._align_margin, self.plib._align_margin + self.EDGE_BINS)
        pred_edge = pred_cdf[..., edge_slice]
        true_edge = true_cdf[..., edge_slice]
        edge_diff = (pred_edge - true_edge) * bright.unsqueeze(-1).float()
        n_edge = bright.sum().item() * self.EDGE_BINS
        if n_edge > 0:
            cdf_rmse_edge = torch.sqrt((edge_diff ** 2).sum() / n_edge).item()
        else:
            cdf_rmse_edge = float("nan")

        rel_vis = torch.abs(pred_vis - true_vis) / (true_vis + 1e-10)
        rel_vis = rel_vis[bright]
        vis_median_rel = rel_vis.median().item() if rel_vis.numel() > 0 else float("nan")
        vis_p90_rel = torch.quantile(rel_vis.float(), 0.9).item() if rel_vis.numel() > 0 else float("nan")

        t0_err = (pred_t0 - true_t0).float()[bright]
        t0_mae = t0_err.abs().mean().item() if t0_err.numel() > 0 else float("nan")
        t0_rmse = torch.sqrt((t0_err ** 2).mean()).item() if t0_err.numel() > 0 else float("nan")

        return {
            "cdf_rmse_full": cdf_rmse_full,
            "cdf_rmse_edge": cdf_rmse_edge,
            "vis_median_rel_err": vis_median_rel,
            "vis_p90_rel_err": vis_p90_rel,
            "t0_mae_bins": t0_mae,
            "t0_rmse_bins": t0_rmse,
        }

    def plot_cdf(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        voxel_idx: int = 0,
        pmt_indices: Optional[Sequence[int]] = None,
        n_bins: int = 1000,
        n_show_bins: int = 80,
    ) -> plt.Figure:
        """Plot true vs predicted CDFs for one voxel, selected PMTs, zoomed on rising edge."""
        pred_cdf = self.plib.reconstruct_cdf(
            pred["coeffs"][voxel_idx : voxel_idx + 1],
            pred["t0"][voxel_idx : voxel_idx + 1],
            n_bins=n_bins,
        ).squeeze(0)
        true_cdf = self.plib.reconstruct_cdf(
            target["coeffs"][voxel_idx : voxel_idx + 1],
            target["t0"][voxel_idx : voxel_idx + 1],
            n_bins=n_bins,
        ).squeeze(0)
        t0_true = target["t0"][voxel_idx]
        if pmt_indices is None:
            pmt_indices = list(range(min(6, pred_cdf.shape[0])))
        n_plots = len(pmt_indices)
        fig, axes = plt.subplots(2, n_plots, figsize=(4 * n_plots, 6), gridspec_kw={"height_ratios": [3, 1]})
        if n_plots == 1:
            axes = axes.reshape(-1, 1)
        for col, pmt in enumerate(pmt_indices):
            t0_p = int(t0_true[pmt].item())
            lo = max(0, t0_p - 5)
            hi = min(n_bins, t0_p + n_show_bins)
            x = np.arange(lo, hi)
            axes[0, col].plot(x, true_cdf[pmt].cpu().numpy()[lo:hi], "k-", lw=1.5, label="true")
            axes[0, col].plot(x, pred_cdf[pmt].cpu().numpy()[lo:hi], "r--", lw=1, label="pred")
            axes[0, col].axvline(t0_p, color="gray", ls=":", lw=0.5)
            axes[0, col].set_title(f"PMT {pmt}")
            if col == 0:
                axes[0, col].legend(fontsize=7, frameon=False)
                axes[0, col].set_ylabel("CDF")
            axes[1, col].plot(
                x,
                (pred_cdf[pmt].cpu().numpy()[lo:hi] - true_cdf[pmt].cpu().numpy()[lo:hi]),
                "r",
                lw=0.8,
            )
            axes[1, col].axhline(0, color="gray", lw=0.5)
            axes[1, col].set_xlabel("time bin")
            if col == 0:
                axes[1, col].set_ylabel("residual")
        fig.suptitle(f"CDF comparison (voxel {voxel_idx})", fontsize=12)
        plt.tight_layout()
        return fig

    def plot_visibility(
        self,
        pred_vis: Union[torch.Tensor, np.ndarray],
        true_vis: Union[torch.Tensor, np.ndarray],
        title: str = "Visibility",
    ) -> plt.Figure:
        """y=x log-log scatter of pred vs true visibility."""
        pred_vis = np.asarray(pred_vis).ravel()
        true_vis = np.asarray(true_vis).ravel()
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(true_vis, pred_vis, s=1, alpha=0.1, color="steelblue", rasterized=True)
        vmin = max(true_vis.min(), pred_vis.min(), 1e-2)
        vmax = max(true_vis.max(), pred_vis.max())
        ax.plot([vmin, vmax], [vmin, vmax], "r-", lw=0.8)
        ax.set_xlabel("true visibility")
        ax.set_ylabel("predicted visibility")
        ax.set_xscale("log")
        ax.set_yscale("log")
        rel = np.abs(pred_vis - true_vis) / (true_vis + 1e-10)
        ax.text(0.05, 0.92, f"med |Δ|/v = {np.median(rel):.2%}", transform=ax.transAxes, fontsize=9, va="top")
        ax.set_title(title)
        return fig

    def plot_t0(
        self,
        pred_t0: Union[torch.Tensor, np.ndarray],
        true_t0: Union[torch.Tensor, np.ndarray],
        title: str = "t0 (onset)",
    ) -> plt.Figure:
        """y=x scatter and residual histogram for t0."""
        pred_t0 = np.asarray(pred_t0).ravel()
        true_t0 = np.asarray(true_t0).ravel()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(true_t0, pred_t0, s=1, alpha=0.1, color="steelblue", rasterized=True)
        tmax = max(true_t0.max(), pred_t0.max()) * 1.02
        axes[0].plot([0, tmax], [0, tmax], "r-", lw=0.8)
        axes[0].set_xlabel("true t0 (bins)")
        axes[0].set_ylabel("predicted t0 (bins)")
        axes[0].set_title(title)
        axes[0].set_aspect("equal")
        dt = pred_t0 - true_t0
        axes[1].hist(dt, bins=100, color="steelblue", edgecolor="none", density=True)
        axes[1].axvline(0, color="red", lw=0.8)
        axes[1].set_xlabel("Δt0 = pred − true (bins)")
        axes[1].set_ylabel("density")
        axes[1].set_title(f"residual — median={np.median(dt):.1f}")
        return fig
