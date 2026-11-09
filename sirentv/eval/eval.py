"""Evaluation CLI: routes to eval_pca / eval_quantile by data.dataset.type.

    torchrun --nproc_per_node=4 sirentv/eval/eval.py --config CFG --ckpt CKPT --output OUT.pt [--pmt-ids 0 40 80]
"""
from __future__ import annotations

import argparse
import datetime
import os

import torch
import torch.distributed as dist
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Config path, or a name in config/ (config/archive/ with --archive)")
    parser.add_argument("--archive", action="store_true", default=False,
                        help="Read --config by name from config/archive/")
    parser.add_argument("--output", type=str, default="eval_results.pt", help="output .pt file path")
    parser.add_argument(
        "--pmt-ids", type=int, nargs="+", default=None,
        help="Also build spatial-slice diagnostics and power spectra for these PMTs, e.g. 0 40 80.",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="Checkpoint to evaluate; overrides model.ckpt_file (often null = untrained model).",
    )
    parser.add_argument(
        "--compare", type=str, nargs="+", default=None,
        choices=["pred_vs_compressed", "pred_vs_raw", "compressed_vs_raw"],
        help="PCA only: restrict the comparisons. Default: all three with raw_quantile_plib "
             "configured, else pred_vs_compressed.",
    )
    parser.add_argument(
        "--spectra", action="store_true", default=True,
        help="(kx, ky, kz) power spectra of v/t0/representation, truth vs prediction, per "
             "--pmt-ids PMT (no-op without --pmt-ids). On by default.",
    )
    parser.add_argument("--no-spectra", action="store_false", dest="spectra")
    parser.add_argument(
        "--spectra-components", type=int, nargs="+", default=None,
        help="PCA components or quantile bins for the representation spectra "
             "(default PCA: 0-4; quantile: u ~ 0, .25, .5, .75, 1).",
    )
    parser.add_argument(
        "--spectra-x-margin", type=int, default=20,
        help="Voxel margin from the x walls for the power_spectra_x_far* variants.",
    )
    parser.add_argument(
        "--grad-cache", type=str, default=None,
        help="Gradient-target cache .h5 for target-side grad spectra (used when its rows match "
             "the evaluated set). Default: train.grad_target_cache_file.",
    )
    parser.add_argument(
        "--no-grad-spectra", action="store_true", default=False,
        help="Skip the target-side gradient spectra even if the config names a cache.",
    )
    parser.add_argument(
        "--x-min", type=float, default=None,
        help="Evaluate only voxels with x >= this [mm] (sets data.x_min), e.g. -2160 for the "
             "active volume, leaving out the LUT voxels behind the PMT plane.",
    )
    args = parser.parse_args()

    is_distributed_env = all(k in os.environ for k in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK'])

    if is_distributed_env:
        # Rank 0 alone builds the diagnostics/spectra while the others wait at the final
        # barrier; past NCCL's default 30 min the watchdog SIGABRTs the idle ranks.
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(hours=2))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        print(f"[main] Initialized distributed training: rank {dist.get_rank()}/{dist.get_world_size()}")
    else:
        print("[main] Running in non-distributed mode (single GPU)")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    from sirentv.training.utils import resolve_config

    with open(resolve_config(args.config, args.archive), "r") as f:
        cfg = yaml.safe_load(f)
    if args.x_min is not None:
        cfg.setdefault("data", {})["x_min"] = args.x_min

    # --grad-cache wins over the config's cache; --no-grad-spectra drops both (None = skip)
    grad_cache_file = args.grad_cache or cfg.get("train", {}).get("grad_target_cache_file")
    if args.no_grad_spectra:
        if grad_cache_file:
            print(f"[main] --no-grad-spectra: ignoring grad cache {grad_cache_file}")
        grad_cache_file = None

    dataset_type = cfg.get("data", {}).get("dataset", {}).get("type", "")
    if dataset_type == "CompressedPLibDataset":
        from sirentv.eval.eval_pca import evaluate_pca
        evaluate_pca(cfg, output_file=args.output, pmt_ids=args.pmt_ids, ckpt_file=args.ckpt,
                     comparisons=args.compare, spectra=args.spectra,
                     spectra_components=args.spectra_components,
                     spectra_x_margin=args.spectra_x_margin,
                     grad_cache_file=grad_cache_file)
    elif dataset_type == "QuantilePLibDataset":
        from sirentv.eval.eval_quantile import evaluate_quantile
        evaluate_quantile(
            cfg,
            output_file=args.output,
            pmt_ids=args.pmt_ids,
            ckpt_file=args.ckpt,
            spectra=args.spectra,
            spectra_components=args.spectra_components,
            spectra_x_margin=args.spectra_x_margin,
            grad_cache_file=grad_cache_file,
        )
    else:
        raise ValueError(f"no evaluator for data.dataset.type {dataset_type!r}")

    if is_distributed_env:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
