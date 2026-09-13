"""Shared evaluation building blocks for eval.py / eval_pca.py / eval_quantile.py.

Two halves, kept in one file since both are "eval utilities" but serve different call
sites -- don't conflate them:

- Streaming/distributed pieces (PairwiseBiasAccumulator, ScalarErrorAccumulator,
  first_nonzero_bin, gather_to_rank0): used by the standalone eval_*.py CLI scripts to
  accumulate bias statistics across an entire (possibly multi-GPU) eval dataset. Pipeline-
  specific per-batch logic (fetching pred/target, reconstructing the CDF from whatever
  compressed representation a model uses) stays in each eval_*.py -- this only holds what's
  identical across pipelines.

- Evaluator: an interactive/notebook-style helper for a single batch at a time --
  RMSE-style metrics and matplotlib plots (CDF, visibility, t0). CompressedPLib-specific
  (uses its windowed reconstruct_cdf(coeffs, t0, n_bins) and _align_margin), meant for
  ad-hoc exploration rather than full-dataset evaluation.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt

from sirentv.data.compressed import CompressedPLib


# ---------------------------------------------------------------------------
# Streaming / distributed accumulation (eval_*.py CLI scripts)
# ---------------------------------------------------------------------------


class PairwiseBiasAccumulator:
    """Accumulates per-PMT / per-tick / overall 2*|p-t|/(p+t) bias between a "pred" and a
    "target" quantity (visibility + CDF), matching the formula in `sirentv.analysis.bias`.
    """

    def __init__(self, n_pmts: int, n_ticks: int, threshold: float, device):
        self.threshold = threshold
        self.vis_bias_sum = torch.zeros(n_pmts, device=device)
        self.vis_bias_sq_sum = torch.zeros(n_pmts, device=device)
        self.vis_count = torch.zeros(n_pmts, device=device)

        self.time_bias_sum = torch.zeros(n_ticks, device=device)
        self.time_bias_sq_sum = torch.zeros(n_ticks, device=device)
        self.time_count = torch.zeros(n_ticks, device=device)

        self.overall_vis_bias_sum = torch.tensor(0.0, device=device)
        self.overall_vis_bias_count = torch.tensor(0.0, device=device)
        self.overall_time_bias_sum = torch.tensor(0.0, device=device)
        self.overall_time_bias_count = torch.tensor(0.0, device=device)

    def update(self, pred_v: torch.Tensor, target_v: torch.Tensor, pred_cdf: torch.Tensor, target_cdf: torch.Tensor):
        from sirentv.analysis import bias as compute_bias

        threshold = self.threshold
        target_dict = {"v_linear": target_v, "t_linear": target_cdf}
        pred_dict = {"v_linear": pred_v, "t_linear": pred_cdf}

        vis_masked_count = (target_v > threshold).sum()
        time_masked_count = (target_cdf > threshold).sum()
        self.overall_vis_bias_sum += compute_bias(target_dict, pred_dict, key="v_linear", threshold=threshold) * vis_masked_count
        self.overall_vis_bias_count += vis_masked_count
        self.overall_time_bias_sum += compute_bias(target_dict, pred_dict, key="t_linear", threshold=threshold) * time_masked_count
        self.overall_time_bias_count += time_masked_count

        # per-PMT visibility bias
        vis_mask = target_v > threshold
        p, t = pred_v, target_v
        vis_bias_vals = torch.where(
            vis_mask, 2 * torch.abs(p - t) / (p + t).clamp(min=1e-10), torch.zeros_like(p)
        )
        self.vis_bias_sum += (vis_bias_vals * vis_mask).sum(dim=0)
        self.vis_bias_sq_sum += ((vis_bias_vals ** 2) * vis_mask).sum(dim=0)
        self.vis_count += vis_mask.sum(dim=0).float()

        # per-tick CDF bias
        n = min(self.time_bias_sum.shape[0], pred_cdf.shape[-1], target_cdf.shape[-1])
        pt, tt = pred_cdf[..., :n], target_cdf[..., :n]
        time_mask = tt > threshold
        time_bias_vals = torch.where(
            time_mask, 2 * torch.abs(pt - tt) / (pt + tt).clamp(min=1e-10), torch.zeros_like(pt)
        )
        time_bias_per_tick = time_bias_vals.sum(dim=1) / time_mask.sum(dim=1).clamp(min=1)
        time_mask_any = time_mask.any(dim=1)
        self.time_bias_sum[:n] += (time_bias_per_tick * time_mask_any).sum(dim=0)
        self.time_bias_sq_sum[:n] += ((time_bias_per_tick ** 2) * time_mask_any).sum(dim=0)
        self.time_count[:n] += time_mask_any.sum(dim=0).float()

    def all_reduce(self):
        for t in [
            self.vis_bias_sum, self.vis_bias_sq_sum, self.vis_count,
            self.time_bias_sum, self.time_bias_sq_sum, self.time_count,
            self.overall_vis_bias_sum, self.overall_vis_bias_count,
            self.overall_time_bias_sum, self.overall_time_bias_count,
        ]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    def finalize(self):
        overall_vis_bias = self.overall_vis_bias_sum / self.overall_vis_bias_count.clamp(min=1)
        overall_time_bias = self.overall_time_bias_sum / self.overall_time_bias_count.clamp(min=1)

        vis_bias_mean = self.vis_bias_sum / self.vis_count.clamp(min=1)
        vis_bias_var = (self.vis_bias_sq_sum / self.vis_count.clamp(min=1)) - vis_bias_mean ** 2
        vis_bias_std = torch.sqrt(vis_bias_var.clamp(min=0))
        vis_bias_sem = vis_bias_std / torch.sqrt(self.vis_count.clamp(min=1))

        time_bias_mean = self.time_bias_sum / self.time_count.clamp(min=1)
        time_bias_var = (self.time_bias_sq_sum / self.time_count.clamp(min=1)) - time_bias_mean ** 2
        time_bias_std = torch.sqrt(time_bias_var.clamp(min=0))
        time_bias_sem = time_bias_std / torch.sqrt(self.time_count.clamp(min=1))

        return {
            "overall": {"vis_bias": overall_vis_bias.cpu(), "time_bias": overall_time_bias.cpu()},
            "visibility_bias": {
                "mean": vis_bias_mean.cpu(), "std": vis_bias_std.cpu(),
                "sem": vis_bias_sem.cpu(), "count": self.vis_count.cpu(),
            },
            "time_bias": {
                "mean": time_bias_mean.cpu(), "std": time_bias_std.cpu(),
                "sem": time_bias_sem.cpu(), "count": self.time_count.cpu(),
            },
        }


class ScalarErrorAccumulator:
    """Accumulates mean/std of a masked scalar absolute error (|pred - target|), e.g. t0 onset error."""

    def __init__(self, device):
        self.err_sum = torch.tensor(0.0, device=device)
        self.err_sq_sum = torch.tensor(0.0, device=device)
        self.count = torch.tensor(0.0, device=device)

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        err = torch.abs(pred - target)[mask]
        self.err_sum += err.sum()
        self.err_sq_sum += (err ** 2).sum()
        self.count += mask.sum()

    def all_reduce(self):
        for t in [self.err_sum, self.err_sq_sum, self.count]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    def finalize(self):
        mean = self.err_sum / self.count.clamp(min=1)
        var = (self.err_sq_sum / self.count.clamp(min=1)) - mean ** 2
        std = torch.sqrt(var.clamp(min=0))
        return {"mean": mean.cpu(), "std": std.cpu(), "count": self.count.cpu()}


class BinnedMeanAccumulator:
    """Accumulates the mean (over visible PMTs, then over voxels with any visible PMT) of an
    arbitrary per-(voxel, PMT, bin) quantity -- same two-stage reduction PairwiseBiasAccumulator
    uses for its per-bin bias curve, but for any quantity, not just the bias formula. Used for
    the adjacent-quantile-time spacing needed to estimate the local density f(Q(u)) for a
    density-corrected per-bin Poisson floor -- the visibility mask is bin-independent (a voxel's
    PMT is either visible or not, regardless of bin), so a single scalar count suffices rather
    than a per-bin count.
    """

    def __init__(self, n_bins: int, device):
        self.sum = torch.zeros(n_bins, device=device)
        self.count = torch.zeros(1, device=device)

    def update(self, values: torch.Tensor, vis_mask: torch.Tensor):
        """values: (B, n_pmt, n_bins). vis_mask: (B, n_pmt) bool, visibility mask."""
        mask_f = vis_mask.unsqueeze(-1).float()
        n_visible = vis_mask.sum(dim=1).clamp(min=1).float()
        per_voxel = (values * mask_f).sum(dim=1) / n_visible.unsqueeze(-1)  # (B, n_bins)
        has_any = vis_mask.any(dim=1)  # (B,)
        self.sum += (per_voxel * has_any.unsqueeze(-1).float()).sum(dim=0)
        self.count += has_any.sum().float()

    def all_reduce(self):
        dist.all_reduce(self.sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.count, op=dist.ReduceOp.SUM)

    def finalize(self):
        return (self.sum / self.count.clamp(min=1)).cpu()


def first_nonzero_bin(wvfm: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """First tick index along the last dim where `wvfm > eps`. (..., n_ticks) -> (...,) long.
    Entries with no bin above `eps` get a sentinel of `n_ticks` (caller should mask these out)."""
    n_ticks = wvfm.shape[-1]
    mask = wvfm > eps
    idx = torch.arange(n_ticks, device=wvfm.device).expand_as(wvfm)
    idx_or_sentinel = torch.where(mask, idx, torch.full_like(idx, n_ticks))
    return idx_or_sentinel.min(dim=-1).values


def gather_to_rank0(tensors: dict, world_size: int, rank: int, device) -> dict | None:
    """Gather a dict of same-leading-dim tensors from all ranks to rank 0, concatenated
    along dim 0. Handles ranks contributing different row counts. Returns the gathered
    dict (moved to CPU) on rank 0, None on other ranks. Single-process: just moves to CPU.
    """
    if world_size <= 1:
        return {k: v.cpu() for k, v in tensors.items()}

    local_size = next(iter(tensors.values())).shape[0]
    local_size_t = torch.tensor([local_size], device=device)
    all_sizes = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size_t)
    all_sizes = [int(s.item()) for s in all_sizes]

    gathered = {}
    for key, local_t in tensors.items():
        trailing_shape = local_t.shape[1:]
        bucket = [torch.zeros(sz, *trailing_shape, device=device, dtype=local_t.dtype) for sz in all_sizes] if rank == 0 else None
        # NCCL requires GPU tensors -- local_t may be CPU-resident (callers commonly accumulate
        # per-batch tensors on CPU throughout a long eval loop to avoid sustained GPU memory
        # growth), so move it to device just for this one-shot collective, not for the whole loop
        dist.gather(local_t.to(device).contiguous(), bucket, dst=0)
        if rank == 0:
            gathered[key] = torch.cat(bucket, dim=0).cpu()

    return gathered if rank == 0 else None


def grid_from_positions(positions: np.ndarray, values: np.ndarray):
    """positions: (N, 3), values: (N,) -> (grid, unique_coords, voxel_size, (ix, iy, iz)).
    Regular-grid reshape of scattered (position, scalar) pairs, e.g. from an eval loop that
    iterated over every voxel (shuffle=False, drop_last=False). Same approach as the
    notebook's compute_gradients_from_positions_fast, factored out so it's not duplicated
    per-notebook-cell and can be reused for pred, target, or residual fields alike.
    """
    x_unique = np.unique(positions[:, 0])
    y_unique = np.unique(positions[:, 1])
    z_unique = np.unique(positions[:, 2])
    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)
    dx = x_unique[1] - x_unique[0] if nx > 1 else 1.0
    dy = y_unique[1] - y_unique[0] if ny > 1 else 1.0
    dz = z_unique[1] - z_unique[0] if nz > 1 else 1.0

    ix = np.searchsorted(x_unique, positions[:, 0])
    iy = np.searchsorted(y_unique, positions[:, 1])
    iz = np.searchsorted(z_unique, positions[:, 2])

    grid = np.zeros((nx, ny, nz), dtype=values.dtype)
    grid[ix, iy, iz] = values

    return grid, (x_unique, y_unique, z_unique), (dx, dy, dz), (ix, iy, iz)


def gradient_magnitude_field(positions: np.ndarray, values: np.ndarray):
    """positions: (N, 3), values: (N,) -> (grad_mag_flat (N,), grid, unique_coords, voxel_size).
    Gradient magnitude of a scalar field gridded from scattered positions (e.g. target
    visibility for one PMT) -- used to flag high-frequency/structural regions (PMT surface,
    wires, ...) purely from the data, no detector geometry needed.
    """
    grid, unique_coords, voxel_size, (ix, iy, iz) = grid_from_positions(positions, values)
    gx, gy, gz = np.gradient(grid, *voxel_size)
    grad_mag_grid = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    grad_mag_flat = grad_mag_grid[ix, iy, iz]
    return grad_mag_flat, grad_mag_grid, grid, unique_coords, voxel_size


def find_hot_spots(grad_mag_grid: np.ndarray, n_spots: int = 5, min_separation_voxels: int = 5, invert: bool = False):
    """Top-`n_spots` distinct voxel indices (ix, iy, iz) in `grad_mag_grid`, greedily picked
    with a minimum voxel separation so they don't all cluster on the same feature (e.g. every
    voxel on one wire).

    invert=False (default): highest-gradient first -- structural/high-frequency regions
    (PMT surface, wires, ...).
    invert=True: lowest-gradient first -- smooth/far regions, useful as a baseline to
    compare against the high-gradient spots.
    """
    order = 1 if invert else -1
    flat_idx_sorted = np.argsort(grad_mag_grid, axis=None)[::order]
    coords_sorted = np.array(np.unravel_index(flat_idx_sorted, grad_mag_grid.shape)).T  # (N, 3)

    picked = []
    for coord in coords_sorted:
        if len(picked) >= n_spots:
            break
        if all(np.max(np.abs(coord - p)) >= min_separation_voxels for p in picked):
            picked.append(coord)
    return picked  # list of (ix, iy, iz) index tuples, ranked per `invert`


def extract_slice_window(grid: np.ndarray, center_idx, axis: int, window: int):
    """2D slice through `grid` at `center_idx` along `axis`, cropped to a `window`x`window`
    box centered on the other two coordinates of `center_idx` -- a "magnified" local view,
    since structural features (surfaces, wires) occupy a tiny fraction of the full volume
    and are invisible in a full-extent slice.
    """
    ci = list(center_idx)
    slicer = [slice(None)] * 3
    slicer[axis] = ci[axis]
    full_slice = grid[tuple(slicer)]  # 2D, axes = the other two dims

    other_axes = [d for d in range(3) if d != axis]
    lo = [max(0, ci[d] - window // 2) for d in other_axes]
    hi = [min(full_slice.shape[k], ci[d] + window // 2) for k, d in enumerate(other_axes)]
    return full_slice[lo[0]:hi[0], lo[1]:hi[1]], (lo, hi)


def _window_bias(pred_window: np.ndarray, target_window: np.ndarray, threshold: float = 1e-6) -> float:
    """Mean 2*|p-t|/(p+t) over a 2D slice window, masked where target <= threshold."""
    mask = target_window > threshold
    if not mask.any():
        return float("nan")
    bias = 2 * np.abs(pred_window - target_window) / np.clip(pred_window + target_window, 1e-10, None)
    return float(bias[mask].mean())


def build_spatial_slice_diagnostics(
    positions: np.ndarray, pred_vis: np.ndarray, target_vis: np.ndarray, pmt_id: int,
    n_hot_spots: int = 3, n_smooth_spots: int = 2, window: int = 15,
):
    """positions: (N, 3), pred_vis/target_vis: (N, n_pmt) -- reuses whatever the eval loop
    already gathered for the y=x visibility scatter plot, no extra per-batch computation.

    Finds high-gradient ("hot", structural: PMT surface, wires, ...) and low-gradient
    ("smooth", far/interior) spots in the target visibility field for one PMT, and extracts
    magnified XY/XZ/YZ slice windows of pred, target, and residual (pred-target) around each,
    since structural features are too small a volume fraction to see in a full-extent slice.
    Each spot also gets a scalar mean-bias-in-window per plane, for quick ranking before
    deciding what to actually plot.
    """
    pred_pmt = pred_vis[:, pmt_id]
    target_pmt = target_vis[:, pmt_id]
    residual_pmt = pred_pmt - target_pmt

    grad_mag_flat, grad_mag_grid, target_grid, unique_coords, voxel_size = gradient_magnitude_field(positions, target_pmt)
    pred_grid, _, _, _ = grid_from_positions(positions, pred_pmt)
    residual_grid, _, _, _ = grid_from_positions(positions, residual_pmt)

    hot_spots = find_hot_spots(grad_mag_grid, n_spots=n_hot_spots, invert=False)
    smooth_spots = find_hot_spots(grad_mag_grid, n_spots=n_smooth_spots, invert=True)

    def _slices_for_spot(center_idx):
        out = {}
        for axis, name in enumerate(["x", "y", "z"]):
            pred_sl, bounds = extract_slice_window(pred_grid, center_idx, axis, window)
            target_sl, _ = extract_slice_window(target_grid, center_idx, axis, window)
            resid_sl, _ = extract_slice_window(residual_grid, center_idx, axis, window)
            out[f"slice_{name}"] = {
                "pred": pred_sl, "target": target_sl, "residual": resid_sl, "bounds": bounds,
                "bias": _window_bias(pred_sl, target_sl),
            }
        return out

    def _spot_records(spots):
        return [
            {"index": tuple(int(c) for c in idx), "grad_mag": float(grad_mag_grid[tuple(idx)]), **_slices_for_spot(idx)}
            for idx in spots
        ]

    return {
        "pmt_id": pmt_id,
        "grad_mag_grid": grad_mag_grid,
        "unique_coords": unique_coords,
        "voxel_size": voxel_size,
        "hot_spots": _spot_records(hot_spots),
        "smooth_spots": _spot_records(smooth_spots),
    }


# ---------------------------------------------------------------------------
# Interactive single-batch evaluation (notebook use, CompressedPLib-specific)
# ---------------------------------------------------------------------------


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
