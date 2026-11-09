"""
One-time, offline precomputation of the exact (M-projection Hutchinson) target-side
spatial-gradient Frobenius norm used by gradient supervision (WeightedGradFrobLoss), for
whichever multi-channel key(s) a training config's train.grad_supervision_keys asks for
("coeffs" for CompressedPLibDataset, "quantiles" for QuantilePLibDataset; "v" is always
included too, but is exact for a single projection regardless -- see
compute_grad_frob_target_exact_projected's docstring).

Why this exists: instantiating CompressedPLibDataset/QuantilePLibDataset with
train.grad_supervision_keys set already triggers this computation as a normal part of
__init__ (via _precompute_grad_targets) -- so simply running training (or even just building
the dataset once) already computes it. The problem this script solves is REPEATING that cost
on every run: the source LUT files under /sdf/data/neutrino/pubdata/... are typically
owned by someone else and not writable by whoever's training, so the target can't be cached
INSIDE the LUT file itself. Instead, set train.grad_target_cache_file in the training config
to a path you own, and run this script ONCE (optionally with --device cuda, since the
per-chunk projection matmul is the one part of this computation that can use a GPU -- see
compute_grad_frob_target_exact_projected's `device` argument) to populate that cache file.
Every subsequent training/eval run using the SAME config (or any other config with an
IDENTICAL train.grad_target_cache_file and matching n_photon/xform/n_projections/etc, see
_grad_cache_meta on the dataset classes) then loads the cache directly instead of
recomputing, at ordinary h5-read speed.

The cache covers ALL voxels of the source LUT (not just whatever subset a particular run's
data.max_len/subsample_factor might use), so it's reusable across e.g. a smoke config and its
corresponding full config, as long as every other cache-relevant setting matches.

Usage:
    python -m sirentv.scripts.precompute_grad_frob_targets --config path/to/train_*.yaml \
        [--device cuda] [--cache-file path/to/cache.h5] [--projection-batch-size 10]

--cache-file overrides train.grad_target_cache_file from the config; if neither is set, this
script raises rather than silently computing without caching (that's just running training).

--projection-batch-size overrides train.grad_target_projection_batch_size (default 10 if unset
anywhere). This bounds how many of the n_projections columns are held in memory at once (see
compute_grad_frob_target_exact_projected's docstring) -- n_projections=100 WITHOUT this cap
materializes a ~52GB array for the full dualpca LUT and gets silently OOM-killed (no traceback,
since SIGKILL gives no chance to flush Python's stdout buffer -- which is fully block-, not
line-buffered, whenever stdout isn't a TTY, e.g. piped through Jupyter's `!` or `> file.log`).
If you have headroom to spare, a larger batch means fewer full passes over the source LUT (each
batch re-reads it in full), at the cost of proportionally more memory.
"""

from __future__ import annotations

import argparse
import sys
import time

import yaml

from sirentv.data.builder import build_dataset


def main():
    # force line-buffered stdout even when piped (Jupyter's `!`, `> file.log`, etc.) -- so if
    # this process ever dies unexpectedly (OOM-kill or otherwise), whatever was printed before
    # that point is actually visible rather than lost in an unflushed buffer.
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True, help="Training config YAML (same one used for `sirentv.train`).")
    parser.add_argument("--device", type=str, default=None, help="e.g. 'cuda' or 'cuda:0' -- runs the per-chunk projection matmul on this device. Default: whatever train.grad_target_device is in the config, or CPU.")
    parser.add_argument("--cache-file", type=str, default=None, help="Overrides train.grad_target_cache_file from the config.")
    parser.add_argument("--projection-batch-size", type=int, default=None, help="Overrides train.grad_target_projection_batch_size from the config (default 10). See module docstring.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.setdefault("train", {})
    if args.cache_file is not None:
        train_cfg["grad_target_cache_file"] = args.cache_file
    if args.device is not None:
        train_cfg["grad_target_device"] = args.device
    if args.projection_batch_size is not None:
        train_cfg["grad_target_projection_batch_size"] = args.projection_batch_size

    cache_file = train_cfg.get("grad_target_cache_file")
    grad_keys = train_cfg.get("grad_supervision_keys", [])
    if not grad_keys:
        raise ValueError(
            "train.grad_supervision_keys is empty in this config -- nothing to precompute. "
            "This script only makes sense for a gradient-supervision config."
        )
    if not cache_file:
        raise ValueError(
            "train.grad_target_cache_file is not set (and --cache-file wasn't given) -- "
            "without it, this computation isn't cached anywhere, so running this script "
            "would do the same work training already does, with no benefit. Set "
            "train.grad_target_cache_file in the config or pass --cache-file."
        )

    # keep the rest of dataset construction lightweight -- only the grad-target
    # precompute/cache step (triggered inside __init__ via _precompute_grad_targets) is what
    # this script actually needs; force lazy CPU reads so the (potentially very large) main
    # fields aren't also fully preloaded.
    plib_key = "compressed_plib" if "compressed_plib" in cfg else ("quantile_plib" if "quantile_plib" in cfg else "photonlib")
    cfg.setdefault(plib_key, {})["device"] = "cpu"
    cfg[plib_key]["lazy"] = True

    device_msg = train_cfg.get("grad_target_device") or "cpu"
    batch_msg = train_cfg.get("grad_target_projection_batch_size", 10)
    print(f"[precompute_grad_frob_targets] keys={grad_keys} cache_file={cache_file} "
          f"device={device_msg} projection_batch_size={batch_msg}", flush=True)
    t_start = time.perf_counter()
    build_dataset(cfg)
    print(f"[precompute_grad_frob_targets] done in {time.perf_counter() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
