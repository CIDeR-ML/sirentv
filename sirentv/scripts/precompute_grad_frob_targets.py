"""Fill train.grad_target_cache_file once, offline, for a gradient-supervision config.

Building the dataset computes the target `{key}_grad_frob` anyway; this script does only that,
so every later run with matching cache meta (n_photon, xform, n_projections, ..., see the
datasets' _grad_cache_meta) just reads it. The LUTs themselves are not writable, hence a
separate cache, which covers every LUT voxel and so serves any subsampling.

    python -m sirentv.scripts.precompute_grad_frob_targets --config train_*.yaml \
        [--device cuda] [--cache-file cache.h5] [--projection-batch-size 10]

--device runs the projection matmul on a GPU. --projection-batch-size caps the projections
held at once: 100 at once is ~52 GB for the full PCA LUT and gets SIGKILLed with no output.
Larger batches mean fewer passes over the LUT.
"""

from __future__ import annotations

import argparse
import sys
import time

import yaml

from sirentv.data.builder import build_dataset


def main():
    # line-buffered even when piped, so an OOM-kill doesn't swallow the output so far
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

    # only the grad-target step of dataset construction is needed: keep the main fields lazy
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
