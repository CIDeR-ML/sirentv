"""Shared pieces of the eval_pca.py / eval_quantile.py pipelines: setup, streaming accumulators
that all-reduce across ranks, gathering, and the post-hoc spatial diagnostics."""

from __future__ import annotations

import copy
import os

import numpy as np
import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Setup / output
# ---------------------------------------------------------------------------


def setup_eval_model(cfg: dict, ckpt_file: str | None, tag: str):
    """Load SirenTV (ckpt_file overrides model.ckpt_file) on this rank's device, in eval mode.

    Returns (rank, world_size, device, net_module) with net_module unwrapped from DDP.
    """
    from sirentv.models import SirenTV
    from sirentv.utils.comm import create_ddp_model

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    is_distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    if ckpt_file is not None:
        cfg.setdefault("model", {})["ckpt_file"] = ckpt_file

    net = SirenTV(cfg, weights_only=True).to(device)
    if is_distributed:
        net = create_ddp_model(net, device_ids=[local_rank])
    net.eval()
    net_module = net.module if is_distributed else net

    if rank == 0:
        print(f"[{tag}] Model device: {device}")
        print(f"[{tag}] Distributed: {is_distributed}, world_size: {world_size}")
        if cfg.get("model", {}).get("ckpt_file"):
            print(f"[{tag}] Loaded checkpoint: {cfg['model']['ckpt_file']}")
    return rank, world_size, device, net_module


def eval_loader_cfg(cfg: dict) -> dict:
    """Copy of cfg whose loader visits every voxel once, in a fixed order."""
    eval_cfg = copy.deepcopy(cfg)
    if "data" in eval_cfg and "loader" in eval_cfg["data"]:
        eval_cfg["data"]["loader"]["drop_last"] = False
        eval_cfg["data"]["loader"]["shuffle"] = False
    return eval_cfg


def save_results(results: dict, output_file: str, tag: str):
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(results, output_file)
    print(f"[{tag}] Results saved to {output_file}")


# ---------------------------------------------------------------------------
# Per-batch metrics
# ---------------------------------------------------------------------------


def quantile_spacing(t: torch.Tensor) -> torch.Tensor:
    """Local spacing b_k of a monotone quantile function along the last axis: centred
    difference (t[k+1] - t[k-1]) / 2, one-sided at the first/last bin."""
    b = torch.empty_like(t)
    b[..., 1:-1] = 0.5 * (t[..., 2:] - t[..., :-2])
    b[..., 0] = t[..., 1] - t[..., 0]
    b[..., -1] = t[..., -1] - t[..., -2]
    # No floor: the smallest real spacings (PDF peak, >~4e-4 ns) matter most. Callers must
    # exclude b == 0 (degenerate truth, e.g. one detected photon), where |p-t|/b is undefined.
    return b


def sample_poisson_spacing_err(t: torch.Tensor, n_photons: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """One Monte Carlo draw per (voxel, PMT, bin) of |Q_hat(u_k) - t_k| / b_k: Q_hat is the
    empirical quantile of n_photons photons drawn from the truth quantile function t(u).

    Its order statistic is U ~ Beta(h, N+1-h), h = (N-1)u + 1 (linear-interpolation
    quantile), mapped through t(u); only per-bin marginals are needed, so bins are drawn
    independently. U outside [u_0, u_K-1] is clamped to the end values (truncated tails).
    t: (B, n_pmt, K) linear ns; n_photons: (B, n_pmt) detected counts; u: (K,) sorted.
    """
    N = n_photons.float().clamp(min=2.0).unsqueeze(-1)  # masked-out pairs only; keeps Beta valid
    h = (N - 1) * u + 1
    U = torch.distributions.Beta(h, N + 1 - h).sample()
    idx = torch.searchsorted(u, U.contiguous()).clamp(1, u.numel() - 1)
    u_lo, u_hi = u[idx - 1], u[idx]
    t_lo, t_hi = t.gather(-1, idx - 1), t.gather(-1, idx)
    w = ((U - u_lo) / (u_hi - u_lo)).clamp(0.0, 1.0)
    q_hat = t_lo + w * (t_hi - t_lo)
    b = quantile_spacing(t)
    return torch.abs(q_hat - t) / torch.where(b > 0, b, torch.ones_like(b))


def masked_bin_bias(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B, n_pmt, K) -> (B, n_pmt): mean of 2|p-t|/(p+t) over the masked bins (0 if none)."""
    vals = torch.where(
        mask,
        2 * torch.abs(pred - target) / (pred + target).clamp(min=1e-10),
        torch.zeros_like(pred),
    )
    return vals.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)


def error_correlation(vis_errors: torch.Tensor, quantile_errors: torch.Tensor) -> dict:
    """z-scored per-(voxel, PMT) visibility and quantile errors, and their correlation."""
    vis_err_flat = vis_errors.flatten()
    quantile_err_flat = quantile_errors.flatten()
    vis_err_z = (vis_err_flat - vis_err_flat.mean()) / vis_err_flat.std()
    quantile_err_z = (quantile_err_flat - quantile_err_flat.mean()) / quantile_err_flat.std()
    correlation = torch.corrcoef(torch.stack([vis_err_z, quantile_err_z]))[0, 1]
    return {
        "vis_errors": vis_errors,
        "quantile_errors": quantile_errors,
        "vis_errors_z": vis_err_z,
        "quantile_errors_z": quantile_err_z,
        "correlation": correlation.cpu(),
    }


def first_nonzero_bin(wvfm: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """First tick index along the last dim where `wvfm > eps`. (..., n_ticks) -> (...,) long.
    Entries with no bin above `eps` get a sentinel of `n_ticks` (caller should mask these out)."""
    n_ticks = wvfm.shape[-1]
    mask = wvfm > eps
    idx = torch.arange(n_ticks, device=wvfm.device).expand_as(wvfm)
    idx_or_sentinel = torch.where(mask, idx, torch.full_like(idx, n_ticks))
    return idx_or_sentinel.min(dim=-1).values


# ---------------------------------------------------------------------------
# Streaming accumulators (all_reduce across ranks, finalize on rank 0)
# ---------------------------------------------------------------------------


def _mean_std_sem(total, sq_total, count):
    """count is already clamped to >= 1."""
    mean = total / count
    var = (sq_total / count) - mean ** 2
    std = torch.sqrt(var.clamp(min=0))
    return mean, std, std / torch.sqrt(count)


def _all_reduce(tensors):
    for t in tensors:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)


class PairwiseBiasAccumulator:
    """Per-PMT / per-bin / overall 2|p-t|/(p+t) bias between a pred and a target (visibility + time),
    same formula as `sirentv.analysis.bias`. Per-bin variants: relative bias, absolute error,
    error in units of the local truth spacing b_k, and its Monte Carlo Poisson floor.
    A half never updated with a prediction finalizes to NaN, not a misleading 0.
    """

    def __init__(self, n_pmts: int, n_ticks: int, threshold: float, device):
        self.threshold = threshold
        zeros = lambda n: torch.zeros(n, device=device)  # noqa: E731
        scalar = lambda: torch.tensor(0.0, device=device)  # noqa: E731
        self.vis_bias_sum, self.vis_bias_sq_sum, self.vis_count = zeros(n_pmts), zeros(n_pmts), zeros(n_pmts)

        self.time_bias_sum, self.time_bias_sq_sum, self.time_count = zeros(n_ticks), zeros(n_ticks), zeros(n_ticks)
        self.time_abs_err_sum, self.time_abs_err_sq_sum = zeros(n_ticks), zeros(n_ticks)
        # own count: b_k == 0 bins are excluded on top of the time mask
        self.time_spacing_bias_sum, self.time_spacing_bias_sq_sum = zeros(n_ticks), zeros(n_ticks)
        self.time_spacing_count = zeros(n_ticks)
        self.time_spacing_floor_sum, self.time_spacing_floor_sq_sum = zeros(n_ticks), zeros(n_ticks)

        self.overall_vis_bias_sum, self.overall_vis_bias_count = scalar(), scalar()
        self.overall_time_bias_sum, self.overall_time_bias_count = scalar(), scalar()
        self.overall_time_abs_err_sum = scalar()
        self.overall_time_spacing_bias_sum, self.overall_time_spacing_count = scalar(), scalar()
        self.overall_time_spacing_floor_sum = scalar()

        self._vis_used = False
        self._time_used = False
        self._floor_used = False

    def update(
        self,
        pred_v: torch.Tensor | None,
        target_v: torch.Tensor,
        pred_cdf: torch.Tensor | None,
        target_cdf: torch.Tensor,
        time_pair_mask: torch.Tensor | None = None,
        time_floor_err: torch.Tensor | None = None,
    ):
        """pred_v/pred_cdf None (model has no such branch) skips that half.
        time_pair_mask (B, n_pmts) is ANDed into every time mask on top of target_cdf > threshold.
        time_floor_err (B, n_pmts, n_ticks) is a Poisson-floor sample (sample_poisson_spacing_err),
        reduced like the spacing bias.
        """
        from sirentv.analysis import bias as compute_bias

        threshold = self.threshold

        if pred_v is not None:
            self._vis_used = True
            vis_masked_count = (target_v > threshold).sum()
            self.overall_vis_bias_sum += compute_bias(
                {"v_linear": target_v}, {"v_linear": pred_v}, key="v_linear", threshold=threshold
            ) * vis_masked_count
            self.overall_vis_bias_count += vis_masked_count

            vis_mask = target_v > threshold
            p, t = pred_v, target_v
            vis_bias_vals = torch.where(
                vis_mask, 2 * torch.abs(p - t) / (p + t).clamp(min=1e-10), torch.zeros_like(p)
            )
            self.vis_bias_sum += (vis_bias_vals * vis_mask).sum(dim=0)
            self.vis_bias_sq_sum += ((vis_bias_vals ** 2) * vis_mask).sum(dim=0)
            self.vis_count += vis_mask.sum(dim=0).float()

        if pred_cdf is not None:
            self._time_used = True
            overall_mask = target_cdf > threshold
            if time_pair_mask is not None:
                overall_mask = overall_mask & time_pair_mask[..., None]
            abs_err = torch.abs(pred_cdf - target_cdf)
            rel_err = 2 * abs_err / (pred_cdf + target_cdf).clamp(min=1e-10)
            self.overall_time_bias_sum += torch.where(overall_mask, rel_err, torch.zeros_like(rel_err)).sum()
            self.overall_time_bias_count += overall_mask.sum()
            # spacing from the full-length target, so the end bins get their one-sided difference
            b_k = quantile_spacing(target_cdf)
            spacing_mask_full = overall_mask & (b_k > 0)
            spacing_err = torch.where(
                spacing_mask_full, abs_err / torch.where(b_k > 0, b_k, torch.ones_like(b_k)), torch.zeros_like(abs_err)
            )
            self.overall_time_abs_err_sum += torch.where(overall_mask, abs_err, torch.zeros_like(abs_err)).sum()
            self.overall_time_spacing_bias_sum += spacing_err.sum()
            self.overall_time_spacing_count += spacing_mask_full.sum()
            if time_floor_err is not None:
                self._floor_used = True
                floor_err = torch.where(spacing_mask_full, time_floor_err, torch.zeros_like(time_floor_err))
                self.overall_time_spacing_floor_sum += floor_err.sum()

            # per-bin: mean over PMTs, then over voxels with any masked PMT
            n = min(self.time_bias_sum.shape[0], pred_cdf.shape[-1], target_cdf.shape[-1])
            pt = pred_cdf[..., :n]
            time_mask = overall_mask[..., :n]
            time_bias_vals = torch.where(time_mask, rel_err[..., :n], torch.zeros_like(pt))
            time_abs_err_vals = torch.where(time_mask, abs_err[..., :n], torch.zeros_like(pt))
            spacing_mask = spacing_mask_full[..., :n]
            n_masked_pmts = time_mask.sum(dim=1).clamp(min=1)
            time_bias_per_tick = time_bias_vals.sum(dim=1) / n_masked_pmts
            time_abs_err_per_tick = time_abs_err_vals.sum(dim=1) / n_masked_pmts
            time_spacing_per_tick = spacing_err[..., :n].sum(dim=1) / spacing_mask.sum(dim=1).clamp(min=1)
            spacing_mask_any = spacing_mask.any(dim=1)
            time_mask_any = time_mask.any(dim=1)
            self.time_bias_sum[:n] += (time_bias_per_tick * time_mask_any).sum(dim=0)
            self.time_bias_sq_sum[:n] += ((time_bias_per_tick ** 2) * time_mask_any).sum(dim=0)
            self.time_abs_err_sum[:n] += (time_abs_err_per_tick * time_mask_any).sum(dim=0)
            self.time_abs_err_sq_sum[:n] += ((time_abs_err_per_tick ** 2) * time_mask_any).sum(dim=0)
            self.time_spacing_bias_sum[:n] += (time_spacing_per_tick * spacing_mask_any).sum(dim=0)
            self.time_spacing_bias_sq_sum[:n] += ((time_spacing_per_tick ** 2) * spacing_mask_any).sum(dim=0)
            self.time_spacing_count[:n] += spacing_mask_any.sum(dim=0).float()
            if time_floor_err is not None:
                floor_per_tick = floor_err[..., :n].sum(dim=1) / spacing_mask.sum(dim=1).clamp(min=1)
                self.time_spacing_floor_sum[:n] += (floor_per_tick * spacing_mask_any).sum(dim=0)
                self.time_spacing_floor_sq_sum[:n] += ((floor_per_tick ** 2) * spacing_mask_any).sum(dim=0)
            self.time_count[:n] += time_mask_any.sum(dim=0).float()

    def all_reduce(self):
        _all_reduce([
            self.vis_bias_sum, self.vis_bias_sq_sum, self.vis_count,
            self.time_bias_sum, self.time_bias_sq_sum, self.time_count,
            self.time_abs_err_sum, self.time_abs_err_sq_sum,
            self.time_spacing_bias_sum, self.time_spacing_bias_sq_sum, self.time_spacing_count,
            self.time_spacing_floor_sum, self.time_spacing_floor_sq_sum,
            self.overall_time_spacing_floor_sum,
            self.overall_vis_bias_sum, self.overall_vis_bias_count,
            self.overall_time_bias_sum, self.overall_time_bias_count,
            self.overall_time_abs_err_sum,
            self.overall_time_spacing_bias_sum, self.overall_time_spacing_count,
        ])

    def finalize(self):
        nan = torch.tensor(float("nan"))
        nan_pmt = torch.full_like(self.vis_bias_sum, float("nan")).cpu()
        nan_tick = torch.full_like(self.time_bias_sum, float("nan")).cpu()

        if not self._vis_used:
            overall_vis_bias = nan
            vis_bias_mean = vis_bias_std = vis_bias_sem = nan_pmt
        else:
            overall_vis_bias = self.overall_vis_bias_sum / self.overall_vis_bias_count.clamp(min=1)
            vis_bias_mean, vis_bias_std, vis_bias_sem = _mean_std_sem(
                self.vis_bias_sum, self.vis_bias_sq_sum, self.vis_count.clamp(min=1))

        if not self._time_used:
            overall_time_bias = overall_time_abs_err = overall_time_spacing_bias = nan
            time_bias_mean = time_bias_std = time_bias_sem = nan_tick
            time_abs_err_mean = time_abs_err_std = time_abs_err_sem = nan_tick
            time_spacing_mean = time_spacing_std = time_spacing_sem = nan_tick
        else:
            count = self.time_count.clamp(min=1)
            overall_time_bias = self.overall_time_bias_sum / self.overall_time_bias_count.clamp(min=1)
            time_bias_mean, time_bias_std, time_bias_sem = _mean_std_sem(
                self.time_bias_sum, self.time_bias_sq_sum, count)
            overall_time_abs_err = self.overall_time_abs_err_sum / self.overall_time_bias_count.clamp(min=1)
            time_abs_err_mean, time_abs_err_std, time_abs_err_sem = _mean_std_sem(
                self.time_abs_err_sum, self.time_abs_err_sq_sum, count)
            overall_time_spacing_bias = self.overall_time_spacing_bias_sum / self.overall_time_spacing_count.clamp(min=1)
            time_spacing_mean, time_spacing_std, time_spacing_sem = _mean_std_sem(
                self.time_spacing_bias_sum, self.time_spacing_bias_sq_sum, self.time_spacing_count.clamp(min=1))

        if not (self._time_used and self._floor_used):
            overall_time_spacing_floor = nan
            floor_mean = floor_std = floor_sem = nan_tick
        else:
            overall_time_spacing_floor = self.overall_time_spacing_floor_sum / self.overall_time_spacing_count.clamp(min=1)
            floor_mean, floor_std, floor_sem = _mean_std_sem(
                self.time_spacing_floor_sum, self.time_spacing_floor_sq_sum, self.time_spacing_count.clamp(min=1))

        def stats(mean, std, sem, count):
            return {"mean": mean.cpu(), "std": std.cpu(), "sem": sem.cpu(), "count": count.cpu()}

        return {
            "overall": {
                "vis_bias": overall_vis_bias.cpu(),
                "time_bias": overall_time_bias.cpu(),
                "time_abs_error": overall_time_abs_err.cpu(),
                "time_spacing_bias": overall_time_spacing_bias.cpu(),
                "time_spacing_floor": overall_time_spacing_floor.cpu(),
            },
            "visibility_bias": stats(vis_bias_mean, vis_bias_std, vis_bias_sem, self.vis_count),
            "time_bias": stats(time_bias_mean, time_bias_std, time_bias_sem, self.time_count),
            "time_abs_error": stats(time_abs_err_mean, time_abs_err_std, time_abs_err_sem, self.time_count),
            "time_spacing_bias": stats(time_spacing_mean, time_spacing_std, time_spacing_sem, self.time_spacing_count),
            "time_spacing_floor": stats(floor_mean, floor_std, floor_sem, self.time_spacing_count),
        }


class ScalarErrorAccumulator:
    """Mean/std of a masked absolute error |pred - target|, e.g. the t0 onset error (NaN if never updated)."""

    def __init__(self, device):
        self.err_sum = torch.tensor(0.0, device=device)
        self.err_sq_sum = torch.tensor(0.0, device=device)
        self.count = torch.tensor(0.0, device=device)
        self._used = False

    def update(self, pred: torch.Tensor | None, target: torch.Tensor, mask: torch.Tensor):
        if pred is None:
            return
        self._used = True
        err = torch.abs(pred - target)[mask]
        self.err_sum += err.sum()
        self.err_sq_sum += (err ** 2).sum()
        self.count += mask.sum()

    def all_reduce(self):
        _all_reduce([self.err_sum, self.err_sq_sum, self.count])

    def finalize(self):
        if not self._used:
            nan = torch.tensor(float("nan"))
            return {"mean": nan, "std": nan, "count": self.count.cpu()}
        mean, std, _ = _mean_std_sem(self.err_sum, self.err_sq_sum, self.count.clamp(min=1))
        return {"mean": mean.cpu(), "std": std.cpu(), "count": self.count.cpu()}


class BinnedMeanAccumulator:
    """Mean of a per-(voxel, PMT, bin) quantity over visible PMTs, then over voxels with any
    visible PMT (the visibility mask is bin-independent, so one count suffices). Used for the
    adjacent-quantile spacing behind a density-corrected Poisson floor."""

    def __init__(self, n_bins: int, device):
        self.sum = torch.zeros(n_bins, device=device)
        self.count = torch.zeros(1, device=device)

    def update(self, values: torch.Tensor, vis_mask: torch.Tensor):
        """values: (B, n_pmt, n_bins). vis_mask: (B, n_pmt) bool."""
        mask_f = vis_mask.unsqueeze(-1).float()
        n_visible = vis_mask.sum(dim=1).clamp(min=1).float()
        per_voxel = (values * mask_f).sum(dim=1) / n_visible.unsqueeze(-1)  # (B, n_bins)
        has_any = vis_mask.any(dim=1)  # (B,)
        self.sum += (per_voxel * has_any.unsqueeze(-1).float()).sum(dim=0)
        self.count += has_any.sum().float()

    def all_reduce(self):
        _all_reduce([self.sum, self.count])

    def finalize(self):
        return (self.sum / self.count.clamp(min=1)).cpu()


class VisStatsAccumulator:
    """Per-PMT mean target visibility over v > threshold."""

    def __init__(self, n_pmts: int, threshold: float, device):
        self.threshold = threshold
        self.sum = torch.zeros(n_pmts, device=device)
        self.count = torch.zeros(n_pmts, device=device)

    def update(self, v: torch.Tensor) -> torch.Tensor:
        """Returns the (B, n_pmts) mask v > threshold."""
        mask = v > self.threshold
        masked = torch.where(mask, v, torch.zeros_like(v))
        self.sum += (masked * mask).sum(dim=0)
        self.count += mask.sum(dim=0).float()
        return mask

    def all_reduce(self):
        _all_reduce([self.sum, self.count])

    def finalize(self):
        mean = self.sum / self.count.clamp(min=1)
        return {"mean": mean.cpu()}


def gather_to_rank0(tensors: dict, world_size: int, rank: int, device) -> dict | None:
    """Concatenate a dict of same-leading-dim tensors from all ranks (row counts may differ) on rank 0.

    Returns the CPU dict on rank 0, None elsewhere; single-process just moves to CPU.
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
        # NCCL needs GPU tensors; callers keep their per-batch lists on CPU
        dist.gather(local_t.to(device).contiguous(), bucket, dst=0)
        if rank == 0:
            gathered[key] = torch.cat(bucket, dim=0).cpu()

    return gathered if rank == 0 else None


# ---------------------------------------------------------------------------
# Spatial diagnostics (rank 0, on the gathered full-volume arrays)
# ---------------------------------------------------------------------------


def grid_from_positions(positions: np.ndarray, values: np.ndarray):
    """positions (N, 3), values (N,) -> (grid, unique_coords, voxel_size, (ix, iy, iz)).

    Regular-grid reshape of scattered values from an eval over every voxel.
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
    """positions (N, 3), values (N,) -> (grad_mag_flat (N,), grad_mag_grid, grid, unique_coords, voxel_size).

    Flags structural/high-frequency regions (PMT surface, wires, ...) from the data alone.
    """
    grid, unique_coords, voxel_size, (ix, iy, iz) = grid_from_positions(positions, values)
    gx, gy, gz = np.gradient(grid, *voxel_size)
    grad_mag_grid = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    grad_mag_flat = grad_mag_grid[ix, iy, iz]
    return grad_mag_flat, grad_mag_grid, grid, unique_coords, voxel_size


def find_hot_spots(grad_mag_grid: np.ndarray, n_spots: int = 5, min_separation_voxels: int = 5, invert: bool = False):
    """Top-`n_spots` voxel indices (ix, iy, iz), greedily at least `min_separation_voxels` apart.

    invert=False: highest gradient first (structure); invert=True: lowest first (smooth baseline).
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
    return picked


def extract_slice_window(grid: np.ndarray, center_idx, axis: int, window: int):
    """2D slice of `grid` through `center_idx` along `axis`, cropped to window x window around it."""
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
    """Magnified x/y/z slice windows of pred, target and residual around the highest- and
    lowest-gradient spots of one PMT's target visibility, each with its window bias.

    positions (N, 3), pred_vis/target_vis (N, n_pmt), as gathered by the eval loop.
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


def build_power_spectra(
    positions: np.ndarray,
    fields: dict,
    valid: np.ndarray,
    hann: bool = True,
    n_bins: int = 50,
    inpaint: bool = True,
    x_margin_voxels: int | None = None,
    x_margin_mode: str = "either",
):
    """(kx, ky, kz) power spectra of truth and prediction for one PMT.

    fields: {name: {"pred": arr, "target": arr}}, each (N,) or (N, C) for this PMT; either side
        may be missing (e.g. a target-only gradient cache).
    valid:  (N,) target-side mask, applied to the prediction too so untrained output at invalid
        voxels adds no structure. inpaint=False zero-fills them instead of inpainting.
    x_margin_voxels: also drop voxels within this many cells of both x walls ("either") or
        only the PMT wall ("pmt_wall").
    Returns {name: {"pred"|"target": {axis: (k, power)}}, "_grid": {...}}, all numpy.
    FFT conventions live in sirentv.utils.spectrum.
    """
    from sirentv.utils.spectrum import GridIndex, field_spectra

    grid = GridIndex(positions)
    valid = np.asarray(valid, dtype=bool)
    if x_margin_voxels is not None:
        if x_margin_mode == "either":
            valid = valid & grid.x_far_mask(x_margin_voxels)
        elif x_margin_mode == "pmt_wall":
            valid = valid & grid.x_far_pmt_wall_mask(x_margin_voxels)
        else:
            raise ValueError(f"unknown x_margin_mode {x_margin_mode!r}, expected 'either' or 'pmt_wall'")

    out = {}
    for name, sides in fields.items():
        entry = {}
        for side in ("target", "pred"):
            if sides.get(side) is None:
                continue
            entry[side] = field_spectra(
                grid, {name: sides[side]}, valid=valid, hann=hann, n_bins=n_bins, inpaint=inpaint
            )[name]
        out[name] = entry

    out["_grid"] = {
        "shape": grid.shape,
        "spacings": grid.spacings,
        "n_valid": int(np.count_nonzero(valid)),
        "inpaint": inpaint,
        "x_margin_voxels": x_margin_voxels,
    }
    return out


def spectra_variants(x_margin: int) -> dict:
    """{result key: build_power_spectra kwargs} for the four spectra each eval writes."""
    return {
        "power_spectra": {},
        f"power_spectra_x_far{x_margin}": {"x_margin_voxels": x_margin, "x_margin_mode": "either"},
        f"power_spectra_x_far_pmtwall{x_margin}": {"x_margin_voxels": x_margin, "x_margin_mode": "pmt_wall"},
        "power_spectra_zerofill": {"inpaint": False},
    }


def add_power_spectra(results, positions_np, pmt_ids, fields_for_pmt, components, x_margin, tag):
    """results[variant][pid] = build_power_spectra(...) for every variant and PMT.

    fields_for_pmt(i, pid) -> (fields, valid) for the i-th requested PMT.
    """
    variants = spectra_variants(x_margin)
    for variant_key in variants:
        results[variant_key] = {}
    for i, pid in enumerate(pmt_ids):
        print(f"[{tag}] Building power spectra for PMT {pid}...")
        fields, valid = fields_for_pmt(i, pid)
        for variant_key, variant_kwargs in variants.items():
            spec = build_power_spectra(positions_np, fields, valid, **variant_kwargs)
            spec["_components"] = list(components)
            results[variant_key][pid] = spec


def load_grad_targets(grad_cache_file, n_rows, rep_key, tag):
    """{"grad_v", f"grad_{rep_key}"} from a grad-target cache, or None if absent or its rows
    don't match the evaluated set (the cache covers the full LUT)."""
    if not grad_cache_file:
        return None
    import h5py

    with h5py.File(grad_cache_file, "r") as gf:
        if gf["v"].shape[0] != n_rows:
            print(f"[{tag}] grad cache has {gf['v'].shape[0]} rows but eval "
                  f"covered {n_rows} -- skipping gradient spectra")
            return None
        return {"grad_v": gf["v"][:], f"grad_{rep_key}": gf[rep_key][:]}
