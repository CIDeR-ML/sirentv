"""Pieces shared by CompressedPLibDataset and QuantilePLibDataset."""

import numpy as np
import torch


def select_voxel_indices(data_cfg, n_total, pos, rank, tag):
    """LUT voxel ids this dataset serves: the first data.max_len, every data.subsample_factor-th,
    and with data.x_min [mm] only x >= x_min (e.g. -2160 drops the voxels behind the PMT plane
    at x = -2170 mm). pos: (n_total, 3) voxel positions."""
    max_len = data_cfg.get("max_len", -1)
    if max_len is None or max_len < 0:
        max_len = n_total
    effective = min(n_total, max_len)
    subsample_factor = data_cfg.get("subsample_factor", 1)
    if subsample_factor > 1:
        indices = torch.arange(effective, dtype=torch.long)[::subsample_factor]
    else:
        indices = torch.arange(effective, dtype=torch.long)
    x_min = data_cfg.get("x_min")
    if x_min is not None:
        keep = torch.from_numpy(pos[indices.numpy(), 0] >= float(x_min))
        if rank == 0:
            print(f"[{tag}] data.x_min = {float(x_min):g} mm: keeping "
                  f"{int(keep.sum())} of {len(indices)} voxels")
        indices = indices[keep]
    return indices


def grad_target_options(train_cfg):
    """train.grad_target_* / grad_supervision_aggregate options, with their defaults."""
    return {
        # Rademacher projections for multi-channel keys; v (one channel) is exact regardless
        "n_projections": int(train_cfg.get("grad_target_n_projections", 10)),
        # voxels per chunked read; None = one chunk
        "chunk_size": train_cfg.get("grad_target_chunk_size", 200000),
        # must match the model side (SirenTV.forward grad_aggregate)
        "aggregate": bool(train_cfg.get("grad_supervision_aggregate", False)),
        "agg_chunk_size": train_cfg.get("grad_target_agg_chunk_size", 20000),
    }


def precompute_grad_targets(lib, idx, grad_keys, rep_key, read_v, read_rep, n_rep_channels,
                            train_cfg, opts, meta, n_photon, tag):
    """Exact target `{key}_grad_frob` (squared Frobenius norm) for the dataset's voxels `idx`.

    Computed once at native grid resolution over ALL voxels of the LUT (so one cache serves any
    subsampling), via chunked random projections (compute_grad_frob_target_exact_projected), and
    cached to train.grad_target_cache_file keyed on `meta`
    (see sirentv/scripts/precompute_grad_frob_targets.py to fill it offline).
    read_v / read_rep(chunk_idx) return (chunk, n_pmt, channels) arrays for those LUT rows.
    Returns {key: (len(idx), n_pmt)}, or (len(idx),) visibility-weighted sums with aggregate.
    """
    from sirentv.utils.misc import (
        compute_grad_frob_target_exact_projected, load_grad_frob_cache, save_grad_frob_cache,
    )

    n_pmt = lib._n_pmts
    total_voxels = len(lib)
    cache_file = train_cfg.get("grad_target_cache_file")

    full_targets = load_grad_frob_cache(cache_file, grad_keys, meta)
    if full_targets is not None:
        print(f"[{tag}] loaded EXACT target spatial-gradient for {grad_keys} from cache: {cache_file}")
    else:
        full_idx = np.arange(total_voxels)
        pos_np = lib.pos[full_idx]

        # Two validity masks, on purpose:
        # - per (voxel, PMT), for the representation: a zero-visibility pair has no waveform,
        #   so its coefficients/quantiles are meaningless and must not be a finite-difference neighbour;
        # - any PMT, for v: zero visibility to one PMT is real (shadow); only zero to EVERY PMT
        #   (inside a PMT body) is undefined.
        n = total_voxels
        chunk_size = opts["chunk_size"] if opts["chunk_size"] is not None else n
        valid_per_pmt = np.empty((n, n_pmt), dtype=bool)
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            vis_chunk = lib[full_idx[s:e]]["vis"].numpy()
            valid_per_pmt[s:e] = vis_chunk > 0
        valid_agg = valid_per_pmt.any(axis=1)

        device = train_cfg.get("grad_target_device")
        # caps the (N, n_pmt, batch) projected array; 100 projections at once is tens of GB
        projection_batch_size = train_cfg.get("grad_target_projection_batch_size", 10)
        full_targets = {}
        if "v" in grad_keys:
            print(f"[{tag}] computing EXACT target spatial-gradient (v) ...")
            full_targets["v"] = compute_grad_frob_target_exact_projected(
                pos_np, lambda c: read_v(full_idx[c]), n_pmt=n_pmt, n_channels=1, n_projections=1,
                chunk_size=opts["chunk_size"], valid=valid_agg, device=device,
            )
        if rep_key in grad_keys:
            print(f"[{tag}] computing EXACT target spatial-gradient ({rep_key}) "
                  f"via {opts['n_projections']} random projections ...")
            full_targets[rep_key] = compute_grad_frob_target_exact_projected(
                pos_np, lambda c: read_rep(full_idx[c]), n_pmt=n_pmt, n_channels=n_rep_channels,
                n_projections=opts["n_projections"], chunk_size=opts["chunk_size"],
                valid=valid_per_pmt, device=device,
                projection_batch_size=projection_batch_size,
            )

        if cache_file:
            print(f"[{tag}] caching EXACT target spatial-gradient for {grad_keys} to {cache_file}")
            save_grad_frob_cache(cache_file, full_targets, meta)

    grad_targets = {k: torch.from_numpy(np.asarray(v)[idx]).float() for k, v in full_targets.items()}

    if opts["aggregate"]:
        print(f"[{tag}] aggregating gradient targets across PMTs (visibility-weighted) ...")
        # chunked: even the one-shot "vis" read for every voxel was enough to OOM
        n = len(idx)
        chunk_size = opts["agg_chunk_size"] if opts["agg_chunk_size"] is not None else n
        aggregated = {key: torch.empty(n, dtype=torch.float32) for key in grad_targets}
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            vis_chunk = lib[idx[s:e]]["vis"].float() / n_photon  # (chunk, n_pmt)
            for key, per_pmt in grad_targets.items():
                # already squared: the aggregate is the plain weighted sum (as on the model side)
                aggregated[key][s:e] = (vis_chunk * per_pmt[s:e]).sum(dim=1)
        grad_targets = aggregated
    return grad_targets
