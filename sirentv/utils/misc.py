"""
Misc

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import warnings
from collections import abc
import numpy as np
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