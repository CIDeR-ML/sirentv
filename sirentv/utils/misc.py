"""Spatial-gradient (Frobenius-norm) estimators for gradient supervision, their cache, and t0_mask.

Everything is the SQUARED Frobenius norm sum_k ||d value_k / d position||^2: the Hutchinson
estimate of it is unbiased at any number of projections, its sqrt is not (Jensen).
"""

import json
import os

import h5py
import numpy as np
import torch


def t0_mask(
        n_ticks: int,
        t0: torch.Tensor,
        t_profile: torch.Tensor,
        use_CDF: bool = True,
        steepness: float = 10.0,
        temperature: float = 1.0,
        hard: bool = False,
):
    """softmax(t_profile) (cumsum'd if use_CDF), masked by sigmoid((tick - t0) * steepness / temperature).

    t0 (B, n_pmt), t_profile (B, n_pmt, n_ticks) -> (B, n_pmt, n_ticks).
    hard: straight-through estimator (hard step forward, sigmoid gradient).
    """
    device = t_profile.device
    time_indices = torch.arange(n_ticks, device=device, dtype=torch.float32)[None, None, :]

    logits = (time_indices - t0) * steepness

    if hard:
        soft_mask = torch.sigmoid(logits / temperature)
        hard_mask = (time_indices >= t0).float()
        mask = hard_mask - soft_mask.detach() + soft_mask
    else:
        mask = torch.sigmoid(logits / temperature)

    out = t_profile.softmax(-1)
    if use_CDF:
        out = out.cumsum(-1)
    out = out * mask

    return out


def compute_grad_frob_hutchinson(
    pos_masked: torch.Tensor, output_val: torch.Tensor, n_pmts: int, create_graph: bool = True,
    pmt_ids: list[int] | None = None, n_projections: int = 1,
) -> torch.Tensor:
    """Per-(voxel, PMT) squared Frobenius norm of d output_val / d pos_masked, Hutchinson-estimated.

    One backward pass per PMT (not per channel): for Rademacher v, E||sum_k v_k grad(out_k)||^2
    = sum_k ||grad(out_k)||^2. One projection per call is deliberate: each extra one costs a
    backward pass every step, while a fresh v every step averages over training for free
    (grad_target_n_projections only controls the target side).

    pos_masked: (n_valid, 3) leaf with requires_grad that output_val was computed from.
    output_val: (n_valid, n_pmts[, K]).
    create_graph: True for training (differentiable result); False for read-only use saves memory.
    pmt_ids: restrict the per-PMT loop (result columns follow this order). None = all PMTs.
    n_projections: projections per PMT off the same forward pass (diagnostics).
    Returns (n_valid, n_pmt_sel), or (n_valid, n_pmt_sel, n_projections) when n_projections > 1.
    """
    if output_val.dim() == 2:
        output_val = output_val.unsqueeze(-1)  # (n_valid, n_pmts, 1)

    grad_mags_sq = []
    for pmt_idx in (pmt_ids if pmt_ids is not None else range(n_pmts)):
        pmt_val = output_val[:, pmt_idx]  # (n_valid, K)
        proj_mags = []
        for _ in range(n_projections):
            v = torch.randint(0, 2, pmt_val.shape, device=pmt_val.device, dtype=pmt_val.dtype) * 2 - 1
            proj = (pmt_val * v).sum()
            grad = torch.autograd.grad(
                outputs=proj, inputs=pos_masked, create_graph=create_graph, retain_graph=True,
            )[0]  # (n_valid, 3)
            proj_mags.append(grad.norm(dim=-1)**2)
        grad_mags_sq.append(proj_mags[0] if n_projections == 1 else torch.stack(proj_mags, dim=-1))

    return torch.stack(grad_mags_sq, dim=1)


def compute_grad_frob_hutchinson_aggregate(
    pos_masked: torch.Tensor, output_val: torch.Tensor, vis_weight: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    """Visibility-weighted aggregate over PMTs in ONE backward pass (when per-PMT OOMs): (n_valid,).

    proj = sum_{pmt,k} sqrt(w_pmt) v_{pmt,k} out_{pmt,k} gives
    E||grad proj||^2 = sum_pmt w_pmt sum_k ||grad out_{pmt,k}||^2.
    vis_weight (n_valid, n_pmts) is detached, so the loss can't shrink it instead of matching.
    """
    if output_val.dim() == 2:
        output_val = output_val.unsqueeze(-1)  # (n_valid, n_pmts, 1)

    sqrt_w = vis_weight.detach().clamp(min=0).sqrt().unsqueeze(-1)  # (n_valid, n_pmts, 1)
    v = torch.randint(0, 2, output_val.shape, device=output_val.device, dtype=output_val.dtype) * 2 - 1
    proj = (sqrt_w * v * output_val).sum()
    grad = torch.autograd.grad(outputs=proj, inputs=pos_masked, create_graph=create_graph)[0]  # (n_valid, 3)
    return grad.norm(dim=-1)**2


def _shifted(arr: np.ndarray, axis: int, offset: int, pad_value) -> np.ndarray:
    """result[i] = arr[i + offset] along `axis`, pad_value out of bounds (no wraparound)."""
    result = np.full_like(arr, pad_value)
    n = arr.shape[axis]
    src = [slice(None)] * arr.ndim
    dst = [slice(None)] * arr.ndim
    if offset > 0:
        src[axis] = slice(offset, n)
        dst[axis] = slice(0, n - offset)
    else:
        src[axis] = slice(0, n + offset)
        dst[axis] = slice(-offset, n)
    result[tuple(dst)] = arr[tuple(src)]
    return result


def _masked_finite_diff(grid: np.ndarray, valid: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    """Finite difference of grid (..., K) along `axis` that never reads an invalid neighbour.

    Centred where both neighbours are valid (identical to np.gradient on clean data), one-sided
    toward the single valid neighbour otherwise, 0 at invalid centres or isolated points.
    Array edges count as invalid. valid: (nx, ny, nz), shared across K.
    """
    f_plus = _shifted(grid, axis, 1, np.nan)
    f_minus = _shifted(grid, axis, -1, np.nan)
    v_plus = _shifted(valid, axis, 1, False)[..., None]
    v_minus = _shifted(valid, axis, -1, False)[..., None]
    v_self = valid[..., None]

    can_center = v_self & v_plus & v_minus
    can_forward = v_self & v_plus & ~v_minus
    can_backward = v_self & v_minus & ~v_plus

    with np.errstate(invalid="ignore"):  # NaN padding outside each branch
        out = np.where(can_center, (f_plus - f_minus) / (2 * spacing), 0.0)
        out = np.where(can_forward, (f_plus - grid) / spacing, out)
        out = np.where(can_backward, (grid - f_minus) / spacing, out)
    return out


def compute_grad_frob_target(positions: np.ndarray, values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Exact finite-difference squared Frobenius norm per (voxel, PMT): (N, n_pmt).

    positions (N, 3) on a regular grid, any row order; values (N, n_pmt[, K]).
    valid (N,) or (N, n_pmt): invalid entries are skipped as neighbours (_masked_finite_diff).
    Validity is per (voxel, PMT): a zero-photon pair's coefficients/quantiles are meaningless
    even where the voxel is fine for other PMTs. None = plain np.gradient.
    Loops over PMTs (memory O(N K) per step), vectorized over K.
    """
    if values.ndim == 2:
        values = values[..., None]  # (N, n_pmt, 1)
    n_pmt, K = values.shape[1], values.shape[2]

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

    valid_per_pmt = None
    if valid is not None:
        valid_per_pmt = np.broadcast_to(valid[:, None] if valid.ndim == 1 else valid, (positions.shape[0], n_pmt))

    grad_frob_sq = np.zeros((positions.shape[0], n_pmt), dtype=np.float32)
    for pmt_idx in range(n_pmt):
        grid = np.zeros((nx, ny, nz, K), dtype=np.float64)
        grid[ix, iy, iz, :] = values[:, pmt_idx, :]
        if valid_per_pmt is None:
            gx, gy, gz = np.gradient(grid, dx, dy, dz, axis=(0, 1, 2))
        else:
            valid_grid = np.zeros((nx, ny, nz), dtype=bool)
            valid_grid[ix, iy, iz] = valid_per_pmt[:, pmt_idx]
            gx = _masked_finite_diff(grid, valid_grid, dx, axis=0)
            gy = _masked_finite_diff(grid, valid_grid, dy, axis=1)
            gz = _masked_finite_diff(grid, valid_grid, dz, axis=2)
        grad_mag_sq = gx[ix, iy, iz, :] ** 2 + gy[ix, iy, iz, :] ** 2 + gz[ix, iy, iz, :] ** 2  # (N, K)

        grad_frob_sq[:, pmt_idx] = grad_mag_sq.sum(axis=-1)

    return grad_frob_sq


def compute_grad_frob_target_exact_projected(
    fine_positions: np.ndarray, read_values_fn, n_pmt: int, n_channels: int,
    n_projections: int = 10, seed: int | None = 0, chunk_size: int | None = 200000,
    valid: np.ndarray | None = None, device: str | torch.device | None = None,
    projection_batch_size: int | None = None,
) -> np.ndarray:
    """compute_grad_frob_target for a K-channel field without ever holding (N, n_pmt, K) in memory.

    Each chunk read is projected by one FIXED Rademacher matrix R (K, n_projections) to
    (chunk, n_pmt, M); the finite-difference norm of the projections, divided by M, is an
    unbiased estimate of the squared Frobenius norm at native resolution.
    The target is computed once, so unlike the model side it gets no averaging over steps:
    use several projections. K = 1 (v) is exact with one.

    read_values_fn(chunk_idx) -> (len(chunk_idx), n_pmt, n_channels) raw values.
    chunk_size: voxels per read (None = one read). valid: see compute_grad_frob_target.
    device: run the projection matmul there (e.g. "cuda"); the finite difference stays numpy.
    projection_batch_size: projections held at once; (N, n_pmt, 100) is ~52 GB and gets
        SIGKILLed silently. Batches sum to exactly the same estimate.
    Returns (N, n_pmt) float32.
    """
    rng = np.random.default_rng(seed)
    R_full = (rng.integers(0, 2, size=(n_channels, n_projections)) * 2 - 1).astype(np.float32)  # Rademacher

    use_gpu = device is not None and str(device) != "cpu"

    n = fine_positions.shape[0]
    size = n if chunk_size is None else chunk_size
    batch = n_projections if projection_batch_size is None else min(projection_batch_size, n_projections)

    sum_sq = np.zeros((n, n_pmt), dtype=np.float64)
    n_batches = -(-n_projections // batch)  # ceil div
    for batch_idx, m_start in enumerate(range(0, n_projections, batch)):
        if n_batches > 1:
            print(f"  [compute_grad_frob_target_exact_projected] projection batch "
                  f"{batch_idx + 1}/{n_batches} ...", flush=True)
        m_end = min(m_start + batch, n_projections)
        R = R_full[:, m_start:m_end]
        m_here = m_end - m_start
        R_t = torch.from_numpy(R).to(device) if use_gpu else None

        projected = np.empty((n, n_pmt, m_here), dtype=np.float32)
        for s in range(0, n, size):
            e = min(s + size, n)
            chunk_values = read_values_fn(np.arange(s, e))  # (chunk, n_pmt, K)
            if use_gpu:
                chunk_t = torch.from_numpy(np.asarray(chunk_values, dtype=np.float32)).to(device)
                projected[s:e] = torch.einsum("cpk,km->cpm", chunk_t, R_t).cpu().numpy()
            else:
                projected[s:e] = np.einsum("cpk,km->cpm", chunk_values, R)

        grad_frob_batch = compute_grad_frob_target(fine_positions, projected, valid=valid)  # (N, n_pmt)
        sum_sq += grad_frob_batch.astype(np.float64)

    return (sum_sq / n_projections).astype(np.float32)


# Bump when the MEANING of cached arrays changes in code (config changes are caught by the meta).
GRAD_CACHE_FORMAT_VERSION = 2  # 1: sqrt'd Frobenius-norm magnitude; 2: squared (no sqrt)


def _grad_cache_meta_key(meta: dict) -> str:
    return json.dumps(meta, sort_keys=True)


def load_grad_frob_cache(cache_file: str | None, keys, meta: dict) -> dict | None:
    """{key: (N_total, n_pmt)} for every LUT voxel from `cache_file`, or None if the file is
    missing, a key is absent, or its meta differs from `meta` in any way."""
    if not cache_file or not os.path.exists(cache_file):
        return None
    with h5py.File(cache_file, "r") as f:
        if f.attrs.get("meta_json") != _grad_cache_meta_key(meta):
            return None
        out = {}
        for key in keys:
            if key not in f:
                return None
            out[key] = f[key][:]
        return out


def save_grad_frob_cache(cache_file: str, arrays: dict, meta: dict) -> None:
    """Write full-LUT gradient targets with their meta, overwriting `cache_file`."""
    parent = os.path.dirname(cache_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with h5py.File(cache_file, "w") as f:
        f.attrs["meta_json"] = _grad_cache_meta_key(meta)
        for key, arr in arrays.items():
            f.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
