"""
Misc

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import json
import warnings
from collections import abc
import numpy as np
import h5py
import torch
from importlib import import_module


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersection_and_union(output, target, K, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersection_and_union_gpu(output, target, k, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=k, min=0, max=k - 1)
    area_output = torch.histc(output, bins=k, min=0, max=k - 1)
    area_target = torch.histc(target, bins=k, min=0, max=k - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def make_dirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port





def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(f"custom_imports must be a list but got type {type(imports)}")
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported.")
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f"{imp} failed to import and is ignored.", UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


class DummyClass:
    def __init__(self):
        pass


def t0_mask(
        n_ticks: int,
        t0: torch.Tensor,
        t_profile: torch.Tensor,
        use_CDF: bool = True,
        steepness: float = 10.0,
        temperature: float = 1.0,
        hard: bool = False,
):
    """
    Unified time mask with multiple strategies

    Args:
        n_ticks: number of time ticks
        t0: (B, n_pmt) start times
        t_profile: (B, n_pmt, n_ticks) time profile
        use_CDF: whether to compute cumulative sum
        steepness: steepness factor for sigmoid (higher = sharper)
        hard: if True, use straight-through estimator, else just steepness controlled sigmoid

    Returns:
        out: masked profile (B, n_pmt, n_ticks)
    """
    device = t_profile.device
    time_indices = torch.arange(n_ticks, device=device, dtype=torch.float32)[None, None, :]

    logits = (time_indices - t0) * steepness

    if hard:
        # Soft probabilities for gradient
        soft_mask = torch.sigmoid(logits / temperature)

        # Hard threshold for forward pass
        hard_mask = (time_indices >= t0).float()

        # Straight-through estimator: forward uses hard, backward uses soft
        mask = hard_mask - soft_mask.detach() + soft_mask
    else:
        # Just temperature-scaled sigmoid
        mask = torch.sigmoid(logits / temperature)

    out = t_profile.softmax(-1)
    if use_CDF:
        out = out.cumsum(-1)
    out = out * mask

    return out

def compute_gradients_from_positions(positions, visibility, pmt_idx):

    # Find unique coordinates
    x_unique = np.unique(positions[:, 0])
    y_unique = np.unique(positions[:, 1])
    z_unique = np.unique(positions[:, 2])

    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)

    dx = x_unique[1] - x_unique[0]
    dy = y_unique[1] - y_unique[0]
    dz = z_unique[1] - z_unique[0]

    #print(f"Grid: {nx} x {ny} x {nz}, spacing: ({dx:.4f}, {dy:.4f}, {dz:.4f})")

    # Vectorized index lookup
    ix = np.searchsorted(x_unique, positions[:, 0])
    iy = np.searchsorted(y_unique, positions[:, 1])
    iz = np.searchsorted(z_unique, positions[:, 2])

    # Fill grid
    vis_grid = np.zeros((nx, ny, nz))
    vis_pmt = visibility[:, pmt_idx]
    vis_grid[ix, iy, iz] = vis_pmt

    # Compute gradients
    grad_x_grid, grad_y_grid, grad_z_grid = np.gradient(vis_grid, dx, dy, dz)

    # Extract gradients at original positions
    grad_x_flat = grad_x_grid[ix, iy, iz]
    grad_y_flat = grad_y_grid[ix, iy, iz]
    grad_z_flat = grad_z_grid[ix, iy, iz]
    grad_mag = np.sqrt(grad_x_flat ** 2 + grad_y_flat ** 2 + grad_z_flat ** 2)

    return grad_x_flat, grad_y_flat, grad_z_flat, grad_mag, vis_grid, (nx, ny, nz), (dx, dy, dz)


def predict_gradient_magnitudes(
        pred_vis: torch.Tensor,
        positions: torch.Tensor,
) -> torch.Tensor:
    """
    Compute gradient magnitudes of predicted visibility w.r.t. positions via autograd

    Args:
        pred_vis: (B, n_pmts) predicted visibility values
        positions: (B, 3) input positions - must have requires_grad=True

    Returns:
        grad_mag: (B, n_selected_pmts) gradient magnitudes ||∇vis||
    """
    if not positions.requires_grad:
        positions = positions.requires_grad_(True)

    if not pred_vis.requires_grad:
        raise RuntimeError(
            "pred_vis does not require gradients. "
            "This suggests the model is not using positions to compute visibility. "
            "Check that positions has requires_grad=True before the forward pass."
        )

    n_pmts = pred_vis.shape[1]

    pmt_indices = list(range(n_pmts))

    # Process in chunks to avoid memory buildup
    chunk_size = 5  # Process 5 PMTs at a time
    all_grad_mags = []

    for chunk_start in range(0, len(pmt_indices), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(pmt_indices))
        chunk_indices = pmt_indices[chunk_start:chunk_end]

        chunk_grads = []

        for i, pmt_idx in enumerate(chunk_indices):
            vis_pmt = pred_vis[:, pmt_idx]

            is_last = (chunk_start + i == len(pmt_indices) - 1)

            grad = torch.autograd.grad(
                outputs=vis_pmt.sum(),
                inputs=positions,
                create_graph=False,
                retain_graph=True if (i < len(pmt_indices) - 1) else False,
                allow_unused=True
            )[0]

            if grad is None:
                raise RuntimeError(f"Gradient is None for PMT {pmt_idx}")

            grad_mag = torch.norm(grad, dim=-1)
            chunk_grads.append(grad_mag)

        # Stack this chunk
        chunk_result = torch.stack(chunk_grads, dim=1)
        all_grad_mags.append(chunk_result)

        # Clear chunk memory
        del chunk_grads
        torch.cuda.empty_cache()

    return torch.cat(all_grad_mags, dim=1)


def compute_grad_frob_hutchinson(
    pos_masked: torch.Tensor, output_val: torch.Tensor, n_pmts: int, create_graph: bool = True,
    pmt_ids: list[int] | None = None,
) -> torch.Tensor:
    """Unbiased per-(voxel, PMT) Frobenius-norm gradient magnitude of `output_val` w.r.t.
    `pos_masked`, via one random-projection backward pass per PMT -- NOT per output channel.

    Unlike predict_gradient_magnitudes/BranchedSiren.compute_analytical_gradients above, this
    defaults to create_graph=True: the result is a genuine differentiable quantity, usable as
    a training signal in a loss (those two are create_graph=False and get detached, so a loss
    built on top of them contributes zero gradient to the network -- they're diagnostics only).
    Pass create_graph=False when you only need the VALUES (e.g. a read-only notebook
    diagnostic, not a training step) -- PyTorch retains substantially more graph structure
    per backward pass when create_graph=True (it has to, to support differentiating through
    the result again), so leaving it True for a read-only use is a real, avoidable memory cost
    across n_pmts backward passes, not just a correctness no-op.

    Cost is O(n_pmts), independent of output_val's channel count K, via a Hutchinson trace
    estimator: for a random Rademacher vector v (+-1 entries, same shape as one PMT's output),
    E_v[||sum_k v_k * d(output_k)/d(pos)||^2] = sum_k ||d(output_k)/d(pos)||^2, the true
    Frobenius norm squared -- so one projected backward pass per PMT gives an unbiased estimate
    of that PMT's full-Jacobian norm, instead of needing one backward pass per channel.

    Deliberately uses a SINGLE projection per call, not an M-averaged one like the target side's
    compute_grad_frob_target_exact_projected: there, M projections are essentially free (one
    np.gradient call handles any channel count, so M columns cost barely more than 1); here,
    each additional projection needs its own torch.autograd.grad call, so M projections would
    cost M backward passes EVERY training step -- multiplying exactly the cost this Hutchinson
    trick exists to avoid. Instead, this relies on getting a genuinely fresh random v every
    forward pass, so training naturally averages over many independent random directions across
    steps at zero extra cost -- a different (step-averaged, not call-averaged) route to the same
    statistical benefit, not an oversight. grad_target_n_projections in configs intentionally
    controls only the target-side precompute for this reason.

    Args:
        pos_masked: (n_valid, 3), requires_grad=True, the tensor output_val was actually
            computed from (before any PMT-dimension expand/broadcast).
        output_val: (n_valid, n_pmts) or (n_valid, n_pmts, K) -- model output for one key.
        n_pmts: number of PMTs (output_val.shape[1]).
        pmt_ids: if given, restricts the O(n_pmts) backward-pass loop to only these PMT
            indices -- e.g. a read-only diagnostic that only ever looks at one fixed PMT has
            no reason to pay for the other n_pmts-1 backward passes just to discard them.
            None (default) computes all n_pmts, as training needs every PMT's gradient.

    Returns:
        (n_valid, len(pmt_ids) if pmt_ids else n_pmts) gradient magnitude estimate.
    """
    if output_val.dim() == 2:
        output_val = output_val.unsqueeze(-1)  # (n_valid, n_pmts, 1) -- e.g. visibility, log_t0

    grad_mags_sq = []
    for pmt_idx in (pmt_ids if pmt_ids is not None else range(n_pmts)):
        pmt_val = output_val[:, pmt_idx]  # (n_valid, K)
        v = torch.randint(0, 2, pmt_val.shape, device=pmt_val.device, dtype=pmt_val.dtype) * 2 - 1
        proj = (pmt_val * v).sum()
        grad = torch.autograd.grad(
            outputs=proj, inputs=pos_masked, create_graph=create_graph, retain_graph=True,
        )[0]  # (n_valid, 3)
        grad_mags_sq.append(grad.norm(dim=-1)**2)

    return torch.stack(grad_mags_sq, dim=1)  # (n_valid, len(pmt_ids) or n_pmts)


def compute_grad_frob_hutchinson_aggregate(
    pos_masked: torch.Tensor, output_val: torch.Tensor, vis_weight: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    """Fallback for when compute_grad_frob_hutchinson's O(n_pmts) backward passes OOM: a
    single-backward-pass, visibility-weighted AGGREGATE (across all PMTs at once) Frobenius-norm
    gradient magnitude estimate. O(1) cost regardless of n_pmts or channel count K, at the cost
    of per-PMT granularity -- this gives one gradient-magnitude estimate per voxel, not per
    (voxel, PMT).

    Draws a single Rademacher vector v over ALL (pmt, k) pairs at once, scaled per-PMT by
    sqrt(vis_weight) before projecting:
        proj = sum_{pmt,k} sqrt(w_pmt) * v_{pmt,k} * output_{pmt,k}
    Since v is i.i.d. mean-zero, cross terms vanish in expectation and one backward pass gives:
        E[||d(proj)/d(pos)||^2] = sum_pmt w_pmt * sum_k ||d(output_{pmt,k})/d(pos)||^2
    i.e. an unbiased estimator of the visibility-weighted aggregate Frobenius norm squared.

    Single projection per call, same reasoning as compute_grad_frob_hutchinson above: unlike
    the target side (compute_grad_frob_target_exact_projected), where M projections are ~free,
    each extra projection here needs its own backward pass, so this relies on a fresh v every
    training step rather than averaging M projections within one step. Not wired to
    grad_target_n_projections for this reason -- see compute_grad_frob_hutchinson's docstring.

    Args:
        pos_masked: (n_valid, 3), requires_grad=True, the tensor output_val was computed from.
        output_val: (n_valid, n_pmts) or (n_valid, n_pmts, K) -- model output for one key.
        vis_weight: (n_valid, n_pmts), non-negative -- e.g. predicted linear-domain visibility.
            DETACHED internally: this is a fixed per-forward-pass importance weight, not
            something the loss should be able to reduce by shrinking predicted visibility
            rather than actually matching gradients.
        create_graph: see compute_grad_frob_hutchinson's docstring -- defaults to True (needed
            for training), pass False for a read-only diagnostic to avoid retaining unneeded
            graph structure.

    Returns:
        (n_valid,) aggregate gradient magnitude estimate.
    """
    if output_val.dim() == 2:
        output_val = output_val.unsqueeze(-1)  # (n_valid, n_pmts, 1)

    sqrt_w = vis_weight.detach().clamp(min=0).sqrt().unsqueeze(-1)  # (n_valid, n_pmts, 1)
    v = torch.randint(0, 2, output_val.shape, device=output_val.device, dtype=output_val.dtype) * 2 - 1
    proj = (sqrt_w * v * output_val).sum()
    grad = torch.autograd.grad(outputs=proj, inputs=pos_masked, create_graph=create_graph)[0]  # (n_valid, 3)
    return grad.norm(dim=-1)**2  # (n_valid,)


def _shifted(arr: np.ndarray, axis: int, offset: int, pad_value) -> np.ndarray:
    """arr shifted by `offset` along `axis`; out-of-bounds entries filled with pad_value
    (NOT wraparound, unlike np.roll) -- i.e. result[i] = arr[i+offset] where in-bounds."""
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
    """Finite difference of `grid` (..., K) along `axis`, treating BOTH true array boundaries
    and `valid`-masked-False cells as "no neighbor there" -- same idea, unified: a centered
    difference needs a real neighbor on each side, and neither the edge of the array nor an
    invalid (e.g. zero-visibility, inside-a-PMT's-solid-body) voxel counts as one.

    Uses a centered difference where the center and both neighbors are valid (this is IDENTICAL
    to what plain np.gradient computes when there's no invalid data at all -- same formula, same
    boundary treatment). Falls back to a one-sided difference using whichever single neighbor is
    valid when the other side isn't (skipping the invalid side rather than reading through it --
    the point of this function, versus filling invalid entries with a placeholder value that
    would otherwise corrupt a real neighbor's derivative). Returns 0 where the center itself is
    invalid (whatever value ends up there is moot -- the caller masks it out downstream anyway)
    or where neither neighbor is valid (an isolated point, should not occur for real geometry).

    valid: (nx, ny, nz) boolean, shared across the K axis (validity is a property of the voxel,
    not of any one channel).
    """
    f_plus = _shifted(grid, axis, 1, np.nan)
    f_minus = _shifted(grid, axis, -1, np.nan)
    v_plus = _shifted(valid, axis, 1, False)[..., None]
    v_minus = _shifted(valid, axis, -1, False)[..., None]
    v_self = valid[..., None]

    can_center = v_self & v_plus & v_minus
    can_forward = v_self & v_plus & ~v_minus
    can_backward = v_self & v_minus & ~v_plus

    with np.errstate(invalid="ignore"):  # NaN-padded f_plus/f_minus outside their np.where branch
        out = np.where(can_center, (f_plus - f_minus) / (2 * spacing), 0.0)
        out = np.where(can_forward, (f_plus - grid) / spacing, out)
        out = np.where(can_backward, (grid - f_minus) / spacing, out)
    return out


def compute_grad_frob_target(positions: np.ndarray, values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Target-side (finite-difference) counterpart to compute_grad_frob_hutchinson -- computes
    the exact (not estimated) per-(voxel, PMT) SQUARED Frobenius-norm gradient magnitude
    sum_k ||d(value_k)/d(position)||^2 (no sqrt -- pred/target are compared in this squared
    domain directly, since compute_grad_frob_hutchinson[_aggregate] also returns the squared
    quantity: unbiased via the Hutchinson identity even at a single projection, unlike its sqrt,
    which Jensen's inequality makes systematically biased low). One-time/offline; values is
    already fully materialized in memory.

    positions: (N, 3) voxel positions (regular grid, arbitrary row order).
    values: (N, n_pmt) or (N, n_pmt, K) -- e.g. visibility, PCA coefficients, or quantile-
    function values.
    valid: (N,) or (N, n_pmt) boolean, optional. Validity is fundamentally per-(voxel, PMT),
    NOT just per-voxel: a voxel can have nonzero visibility to some PMTs and exactly zero to
    others (occluded, out of view, etc.), and the coeffs/quantiles derived from a zero-photon
    (voxel, PMT) pair can be NaN or otherwise meaningless EVEN THOUGH that same voxel is
    perfectly fine for a DIFFERENT PMT -- checking only a per-voxel aggregate (e.g.
    vis.sum(axis=1) > 0) misses exactly this case. A (N,) array is broadcast across every PMT
    (only correct if validity truly doesn't vary by PMT, e.g. a purely geometric "inside solid
    material" criterion). Invalid entries are skipped as NEIGHBORS in the finite difference,
    falling back to a one-sided formula using whichever single neighbor IS valid, rather than
    reading through their value and corrupting an adjacent real voxel's derivative. None
    (default): original behavior, plain np.gradient, no masking -- use when values is already
    known-clean.

    Loops over PMTs, but vectorizes the finite-difference across all K channels within each PMT
    via a single np.gradient (or, if valid is given, _masked_finite_diff) call per axis instead
    of one call per channel -- this bounds memory to O(n_voxels * K) per iteration rather than
    O(n_voxels * n_pmt * K) all at once, which matters when K is in the hundreds (the quantile
    family).
    """
    if values.ndim == 2:
        values = values[..., None]  # (N, n_pmt, 1) -- e.g. visibility
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
    """Exact (native-resolution, no coarse-graining) finite-difference target SQUARED gradient
    Frobenius norm for a multi-channel field (K=n_channels per PMT, e.g. 50 PCA coefficients or
    512 quantile bins) -- WITHOUT ever materializing the full (N, n_pmt, K) array (K=50 or K=512
    times N~1.6M voxels times n_pmt=81 is tens of GB; an earlier coarse-subsampling approach
    that avoided this by computing on a coarser grid had a real, confirmed systematic
    truncation-error bias that ensemble-averaging couldn't fix, and has since been removed).

    Applies the same Hutchinson random-projection trick already used on the PREDICTION side
    (compute_grad_frob_hutchinson) to the TARGET side instead: for a FIXED random Rademacher
    matrix R (K, n_projections) -- drawn once, reused for every chunk and every voxel in this
    call -- each chunk's raw (chunk, n_pmt, K) values are projected down to
    (chunk, n_pmt, n_projections) DURING the chunked read, so at most one chunk's K-dimensional
    slice is ever in memory at once; the assembled (N, n_pmt, n_projections) array is smaller
    than the true (N, n_pmt, K) one by exactly K/n_projections. Differentiating this projected
    array with compute_grad_frob_target (which already sums squared-gradient contributions over
    whatever the last axis is) gives sum_m ||grad(proj_m)||^2; since each of the M projections
    independently has E[||grad(proj_m)||^2] = ||J||_F^2 (same identity as the prediction-side
    estimator), dividing by n_projections gives an unbiased estimator of the true SQUARED
    Frobenius norm ||J||_F^2 (no sqrt anywhere in this pipeline -- see compute_grad_frob_target's
    docstring for why: pred/target are compared in the squared domain specifically so this
    estimator stays unbiased even at the prediction side's n_projections=1, unlike its sqrt)
    -- exact at the native grid resolution, with no subsampling truncation-error bias and no
    wall-slab (or any other) special-casing needed, since there's no longer a reason to avoid
    reading the full volume.

    IMPORTANT: n_projections=1 would be a biased, ARBITRARY-DIRECTION estimate baked in for the
    entire training run -- unlike the prediction side, which gets a fresh random projection
    every forward pass and averages over many directions across training, a target computed
    ONCE with a single fixed projection never gets that averaging. n_projections should be
    several (a handful to a few dozen) to control this; for n_channels=1 (e.g. "v"), a single
    projection is exact regardless ((+-1 * x)^2 = x^2, and dividing by n_projections=1 is a
    no-op), so this same function is used for both K=1 and K>1 keys uniformly.

    fine_positions: (N, 3) all voxel positions.
    read_values_fn: callable(chunk_idx: np.ndarray) -> (len(chunk_idx), n_pmt, n_channels) --
        raw (unprojected) values for a CONTIGUOUS chunk of voxel indices; reads every voxel,
        just one chunk at a time.
    n_pmt, n_channels: shape of what read_values_fn returns per chunk (n_channels = K).
    n_projections: number of independent Rademacher projections averaged over (see above).
    seed: RNG seed for the (fixed, reused across all chunks) projection matrix.
    chunk_size: voxels per read_values_fn call. None = one shot (no chunking).
    valid: (N,) or (N, n_pmt) boolean, optional -- forwarded to compute_grad_frob_target.
    Validity is per-(voxel, PMT) in general (a voxel can be valid for one PMT and invalid --
    e.g. zero visibility -- for another), not just per-voxel; invalid entries are skipped as
    NEIGHBORS in the finite difference rather than read through, so they can't corrupt an
    adjacent real voxel's derivative. See compute_grad_frob_target's docstring.
    device: if given (e.g. "cuda"), the per-chunk projection matmul (chunk_values @ R) runs on
    this device via torch instead of numpy on CPU -- the finite-difference step afterwards
    (compute_grad_frob_target) is unaffected, since it's memory-bandwidth-bound vectorized numpy
    rather than FLOP-bound, and doesn't benefit from a GPU the way the projection matmul can for
    large chunk_size * n_pmt * n_channels. None (default): unchanged CPU/numpy behavior. This is
    a one-time, offline-precompute-oriented knob -- see precompute_grad_frob_targets.py.
    projection_batch_size: caps how many of the n_projections columns are held in memory (as
    the assembled (N, n_pmt, batch) "projected" array) AT ONCE. chunk_size only bounds the raw
    K-channel READ; the PROJECTED output below it was, until this parameter existed, always
    materialized for ALL N voxels and ALL n_projections columns simultaneously -- fine at
    n_projections=10 (N~1.6M, n_pmt=81: ~5GB), but n_projections=100 (chosen from the
    ex-junjie.ipynb M-sweep to make the aggregate Jensen bias negligible) blows that up to
    ~52GB and silently OOM-kills the process (no stdout at all: SIGKILL gives Python no chance
    to flush its output buffer, which is fully block- not line-buffered when not attached to a
    TTY, e.g. redirected through Jupyter's `!` or piped to a file). Splitting n_projections into
    batches of at most this size and accumulating the sum of squared-gradient-magnitudes across
    batches (mathematically identical to computing all n_projections at once -- see below) caps
    peak memory to N * n_pmt * projection_batch_size regardless of n_projections. None (default):
    unchanged, single-batch behavior.

    Returns: (N, n_pmt) exact (up to the finite-n_projections estimator variance) target
    SQUARED gradient-Frobenius-norm.
    """
    rng = np.random.default_rng(seed)
    R_full = (rng.integers(0, 2, size=(n_channels, n_projections)) * 2 - 1).astype(np.float32)  # Rademacher

    use_gpu = device is not None and str(device) != "cpu"

    n = fine_positions.shape[0]
    size = n if chunk_size is None else chunk_size
    batch = n_projections if projection_batch_size is None else min(projection_batch_size, n_projections)

    # accumulate SUM of squared per-(voxel, PMT) gradient magnitudes across projection batches --
    # grad_frob_batch (from compute_grad_frob_target) is itself sum_{m in batch} ||grad
    # proj_m||^2, so summing across batches gives exactly sum_{ALL m} ||grad
    # proj_m||^2, identical to what a single all-at-once pass would produce (same double sum,
    # just reordered/regrouped) -- this is a pure memory-shape refactor, not a numerical change.
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


# Bump this whenever the MEANING of the cached arrays changes at the code level (e.g. the
# squared-vs-sqrt'd Frobenius-norm convention switch), as opposed to a config value changing --
# config changes are already caught by the rest of the meta dict, but a code-level convention
# change wouldn't touch any config value at all, and would otherwise produce a silent false
# cache-hit: the on-disk numbers would still "match" the meta while meaning something different
# than what the current code expects. CompressedPLibDataset/QuantilePLibDataset's
# _grad_cache_meta both include this via GRAD_CACHE_FORMAT_VERSION so they can't drift apart.
GRAD_CACHE_FORMAT_VERSION = 2  # 1: sqrt'd Frobenius-norm magnitude; 2: squared (no sqrt)


def _grad_cache_meta_key(meta: dict) -> str:
    """Canonical JSON encoding of a grad-frob cache meta dict, for exact-match validation."""
    return json.dumps(meta, sort_keys=True)


def load_grad_frob_cache(cache_file: str | None, keys, meta: dict) -> dict | None:
    """Loads precomputed per-(voxel, PMT) grad-frob targets from `cache_file`, covering ALL
    voxels of the source LUT (not sliced to any particular run's subsample), if the file exists
    AND its stored meta matches `meta` EXACTLY (a strict JSON string comparison -- any config
    drift at all, e.g. a different n_photon, xform, normalize_coeffs, or n_projections,
    invalidates the cache rather than risking a silent mismatch between what was cached and
    what the current run actually needs). Returns None (caller should fall back to computing)
    if the file is missing, the meta doesn't match, or any requested key isn't present.

    keys: iterable of dataset names to load (e.g. ("v", "coeffs")).
    Returns {key: (N_total, n_pmt) np.ndarray}, or None.
    """
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
    """Writes per-(voxel, PMT) grad-frob targets (covering ALL voxels of the source LUT, not
    sliced to any particular run's subsample) to `cache_file`, for reuse by future runs whose
    meta matches exactly (see load_grad_frob_cache). Overwrites any existing file at that path.
    """
    parent = os.path.dirname(cache_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with h5py.File(cache_file, "w") as f:
        f.attrs["meta_json"] = _grad_cache_meta_key(meta)
        for key, arr in arrays.items():
            f.create_dataset(key, data=arr, compression="gzip", compression_opts=4)