from __future__ import annotations

import argparse
import copy
import os

import h5py
import torch
import torch.distributed as dist
import yaml
from tqdm import tqdm

from sirentv.data.quantile import QuantilePLib, create_quantile_dataloader
from sirentv.models import SirenTV
from sirentv.utils.comm import create_ddp_model
from sirentv.eval.utils import (
    PairwiseBiasAccumulator,
    ScalarErrorAccumulator,
    BinnedMeanAccumulator,
    gather_to_rank0,
    build_spatial_slice_diagnostics,
)


@torch.no_grad()
def evaluate_quantile(cfg: dict, output_file: str = "eval_quantile_results.pt", pmt_ids: list[int] | None = None, ckpt_file: str | None = None):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    is_distributed = world_size > 1

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if ckpt_file is not None:
        cfg.setdefault("model", {})["ckpt_file"] = ckpt_file

    net = SirenTV(cfg, weights_only=True).to(device)
    if is_distributed:
        net = create_ddp_model(net, device_ids=[local_rank])
    net.eval()

    net_module = net.module if is_distributed else net

    if rank == 0:
        print(f"[eval_quantile] Model device: {device}")
        print(f"[eval_quantile] Distributed: {is_distributed}, world_size: {world_size}")
        if cfg.get("model", {}).get("ckpt_file"):
            print(f"[eval_quantile] Loaded checkpoint: {cfg['model']['ckpt_file']}")

    # override drop_last/shuffle for eval (we want all samples, in a fixed order)
    eval_cfg = copy.deepcopy(cfg)
    if "data" in eval_cfg and "loader" in eval_cfg["data"]:
        eval_cfg["data"]["loader"]["drop_last"] = False
        eval_cfg["data"]["loader"]["shuffle"] = False
    dl = create_quantile_dataloader(eval_cfg, rank=rank, world_size=world_size)

    plib_cfg = cfg.get("quantile_plib", cfg.get("photonlib", {}))
    filepath = plib_cfg["filepath"]
    mode = plib_cfg.get("mode", "quantile")
    combine_every_quantile = plib_cfg.get("combine_every_quantile", 1)

    # lazy=True: to_linear_time only needs the lightweight scalar attrs (mode,
    # log_quantile_C), never bulk per-voxel data.
    qlib = QuantilePLib.load(filepath, lazy=True, mode=mode, combine_every_quantile=combine_every_quantile)

    n_pmts = cfg.get("data", {}).get("n_pmt", 81)
    n_quantile = 512 // combine_every_quantile
    threshold = 1e-6

    if rank == 0:
        print(f"[eval_quantile] n_pmts: {n_pmts}, n_quantile: {n_quantile}, mode: {mode}, combine_every_quantile: {combine_every_quantile}")

    # pred vs. target, compared directly in quantile-index bins (native u_grid positions) --
    # not interpolated onto a fixed-time CDF grid, which would distort the comparison since
    # quantile levels are equally spaced in probability, not time.
    acc_pred_vs_target = PairwiseBiasAccumulator(n_pmts, n_quantile, threshold, device)

    # per-PMT target visibility distribution stats
    target_vis_sum = torch.zeros(n_pmts, device=device)
    target_vis_rms_sq_sum = torch.zeros(n_pmts, device=device)
    target_vis_count = torch.zeros(n_pmts, device=device)

    # t0 (onset) bias, restricted to visible PMTs
    t0_acc_pred_vs_target = ScalarErrorAccumulator(device)

    # mean adjacent-quantile-time spacing per bin (target side), for estimating the local
    # density f(Q(u)) needed by a density-corrected per-bin Poisson floor
    spacing_acc = BinnedMeanAccumulator(n_quantile - 1, device)

    log_every = 1

    all_pred_vis = []
    all_target_vis = []
    all_pred_t0 = []
    all_target_t0 = []
    all_positions = []
    all_vis_errors = []
    all_quantile_errors = []
    all_time_bias = []

    for batch_idx, data in enumerate(tqdm(dl, desc=f"Rank {rank} eval", disable=(rank != 0))):
        x = data["position"].contiguous().to(device)
        target = {k: v.contiguous().to(device) for k, v in data["target"].items()}

        pred_out: dict[str, torch.Tensor] = net_module(x)

        # visibility: pred["v"] and target["v"] are in the same xformed domain
        pred_v_linear = net_module._inv_xform_vis(pred_out["v"])  # (B, n_pmts)
        target_v_linear = net_module._inv_xform_vis(target["v"])  # (B, n_pmts)

        # onset time: model predicts log(t0), same units as target["t0"] (both log-space here)
        pred_t0 = torch.exp(pred_out["t0"])  # (B, n_pmts)
        target_t0 = torch.exp(target["t0"])  # (B, n_pmts)

        # zero-visibility (voxel, PMT) pairs have no quantile function -- quantiles is NaN
        # there (see quantile.py's quantiles_mask). Replace with 0 before comparing so NaN
        # can't leak into quantile_error/.mean()/.corrcoef() later; these PMTs are already
        # excluded from the bias accumulators via the visibility threshold below.
        target_quantiles = torch.nan_to_num(target["quantiles"], nan=0.0)
        pred_quantiles = pred_out["quantiles"]

        # invert the log transform (if log_quantile mode) but stay in quantile-index space --
        # no interpolation onto a fixed time grid.
        pred_time = qlib.to_linear_time(pred_quantiles)  # (B, n_pmts, n_quantile)
        target_time = qlib.to_linear_time(target_quantiles)  # (B, n_pmts, n_quantile)

        acc_pred_vs_target.update(pred_v_linear, target_v_linear, pred_time, target_time)

        # compute per-position errors for correlation check
        vis_error = (pred_v_linear - target_v_linear).detach()  # (B, n_pmts)
        quantile_error = (pred_time - target_time).mean(dim=-1).detach()  # (B, n_pmts)
        # accumulator lists are held for the whole eval loop (potentially ~1.6M positions) --
        # move to CPU immediately so they don't sit on GPU for the entire run
        all_vis_errors.append(vis_error.cpu())
        all_quantile_errors.append(quantile_error.cpu())

        # per-(voxel, PMT) time bias, same 2*|p-t|/(p+t) formula and time-value masking as
        # PairwiseBiasAccumulator's per-tick time_bias, but reduced over the quantile-bin axis
        # instead of over voxels/PMTs -- lets the notebook bin the actual bias metric (not the
        # raw signed quantile_error, which can cancel across bins) by spatial region, e.g.
        # distance from the detector wall
        time_bias_mask = target_time > threshold  # (B, n_pmts, n_quantile)
        time_bias_vals = torch.where(
            time_bias_mask,
            2 * torch.abs(pred_time - target_time) / (pred_time + target_time).clamp(min=1e-10),
            torch.zeros_like(pred_time),
        )
        time_bias_per_pmt = time_bias_vals.sum(dim=-1) / time_bias_mask.sum(dim=-1).clamp(min=1)  # (B, n_pmts)
        all_time_bias.append(time_bias_per_pmt.detach().cpu())

        # target visibility distribution stats
        target_vis_mask = target_v_linear > threshold
        target_vis_masked = torch.where(target_vis_mask, target_v_linear, torch.zeros_like(target_v_linear))
        target_vis_sum += (target_vis_masked * target_vis_mask).sum(dim=0)
        target_vis_rms_sq_sum += torch.std((target_vis_masked * target_vis_mask), dim=0)
        target_vis_count += target_vis_mask.sum(dim=0).float()

        # t0 (onset) bias, restricted to visible PMTs
        t0_acc_pred_vs_target.update(pred_t0, target_t0, target_vis_mask)

        # adjacent-quantile-time spacing (target side), for the density-corrected Poisson floor
        target_dt = target_time[..., 1:] - target_time[..., :-1]  # (B, n_pmts, n_quantile-1)
        spacing_acc.update(target_dt, target_vis_mask)

        all_pred_vis.append(pred_v_linear.detach().cpu())
        all_target_vis.append(target_v_linear.detach().cpu())
        all_pred_t0.append(pred_t0.detach().cpu())
        all_target_t0.append(target_t0.detach().cpu())
        all_positions.append(x.detach().cpu())

        if (batch_idx + 1) % log_every == 0 and is_distributed:
            local_running_vis = acc_pred_vs_target.overall_vis_bias_sum / acc_pred_vs_target.overall_vis_bias_count.clamp(min=1)
            local_running_time = acc_pred_vs_target.overall_time_bias_sum / acc_pred_vs_target.overall_time_bias_count.clamp(min=1)

            all_vis = [torch.zeros(1, device=device) for _ in range(world_size)]
            all_time = [torch.zeros(1, device=device) for _ in range(world_size)]
            dist.all_gather(all_vis, local_running_vis.unsqueeze(0))
            dist.all_gather(all_time, local_running_time.unsqueeze(0))

            if rank == 0:
                vis_strs = [f"r{i}:{v.item():.3e}" for i, v in enumerate(all_vis)]
                time_strs = [f"r{i}:{t.item():.3e}" for i, t in enumerate(all_time)]
                #print(f"  [batch {batch_idx+1}] bias_vis: [{', '.join(vis_strs)}] | bias_quantile: [{', '.join(time_strs)}]")
        elif (batch_idx + 1) % log_every == 0 and rank == 0:
            running_vis = acc_pred_vs_target.overall_vis_bias_sum / acc_pred_vs_target.overall_vis_bias_count.clamp(min=1)
            running_time = acc_pred_vs_target.overall_time_bias_sum / acc_pred_vs_target.overall_time_bias_count.clamp(min=1)
            #print(f"  [batch {batch_idx+1}] bias_vis: {running_vis.item():.4e}, bias_quantile: {running_time.item():.4e}")

    local_results = {
        "pred_vis": torch.cat(all_pred_vis, dim=0),
        "target_vis": torch.cat(all_target_vis, dim=0),
        "pred_t0": torch.cat(all_pred_t0, dim=0),
        "target_t0": torch.cat(all_target_t0, dim=0),
        "positions": torch.cat(all_positions, dim=0),
        "vis_errors": torch.cat(all_vis_errors, dim=0),
        "quantile_errors": torch.cat(all_quantile_errors, dim=0),
        "time_bias": torch.cat(all_time_bias, dim=0),
    }

    if is_distributed:
        acc_pred_vs_target.all_reduce()
        dist.all_reduce(target_vis_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(target_vis_rms_sq_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(target_vis_count, op=dist.ReduceOp.SUM)
        t0_acc_pred_vs_target.all_reduce()
        spacing_acc.all_reduce()

    gathered = gather_to_rank0(local_results, world_size, rank, device)

    if rank == 0:
        all_pred_vis_tensor = gathered["pred_vis"]
        all_target_vis_tensor = gathered["target_vis"]
        all_pred_t0_tensor = gathered["pred_t0"]
        all_target_t0_tensor = gathered["target_t0"]
        all_positions_tensor = gathered["positions"]
        all_vis_errors_tensor = gathered["vis_errors"]
        all_quantile_errors_tensor = gathered["quantile_errors"]
        all_time_bias_tensor = gathered["time_bias"]

        stats_pred_vs_target = acc_pred_vs_target.finalize()

        target_vis_mean = target_vis_sum / target_vis_count.clamp(min=1)
        target_vis_std = torch.sqrt(target_vis_rms_sq_sum.clamp(min=0))

        t0_stats_pred_vs_target = t0_acc_pred_vs_target.finalize()
        quantile_time_spacing = spacing_acc.finalize()

        print(f"[eval_quantile] Total positions evaluated: {all_pred_vis_tensor.shape[0]}")
        print(f"[eval_quantile] Overall visibility bias: {stats_pred_vs_target['overall']['vis_bias'].item():.6e}")
        print(f"[eval_quantile] Overall quantile-bin bias: {stats_pred_vs_target['overall']['time_bias'].item():.6e}")
        print(f"[eval_quantile] t0 bias mean: {t0_stats_pred_vs_target['mean'].item():.6e}, std: {t0_stats_pred_vs_target['std'].item():.6e}")

        vis_err_flat = all_vis_errors_tensor.flatten()
        quantile_err_flat = all_quantile_errors_tensor.flatten()
        vis_err_z = (vis_err_flat - vis_err_flat.mean()) / vis_err_flat.std()
        quantile_err_z = (quantile_err_flat - quantile_err_flat.mean()) / quantile_err_flat.std()
        correlation = torch.corrcoef(torch.stack([vis_err_z, quantile_err_z]))[0, 1]
        print(f"[eval_quantile] Visibility-quantile error correlation: {correlation.item():.6f}")

        results = {
            "pred_vs_target": {
                **stats_pred_vs_target,
                "t0_bias": t0_stats_pred_vs_target,
            },
            "target": {
                "visibility_per_pmt": {"mean": target_vis_mean.cpu(), "std": target_vis_std.cpu()},
            },
            "visibility_all": {
                "pred": all_pred_vis_tensor,
                "target": all_target_vis_tensor,
                "positions": all_positions_tensor,
            },
            "t0_all": {
                "pred": all_pred_t0_tensor,
                "target": all_target_t0_tensor,
            },
            "error_correlation": {
                "vis_errors": all_vis_errors_tensor,
                "quantile_errors": all_quantile_errors_tensor,
                "vis_errors_z": vis_err_z,
                "quantile_errors_z": quantile_err_z,
                "correlation": correlation.cpu(),
                # per-(voxel, PMT) time bias (actual 2*|p-t|/(p+t) formula, not the raw signed
                # quantile_errors above) -- for binning the real bias metric spatially, e.g. by
                # distance from the detector wall
                "time_bias": all_time_bias_tensor,
            },
            # mean adjacent-quantile-time spacing per bin (target side), length n_quantile-1 --
            # lets the notebook estimate the local density f(Q(u)) for a density-corrected
            # per-bin Poisson floor: f(Q(u_k)) ~= delta_u / quantile_time_spacing[k]
            "quantile_time_spacing": quantile_time_spacing,
            "meta": {
                "n_pmts": n_pmts,
                "n_quantile": n_quantile,
                "n_positions": all_pred_vis_tensor.shape[0],
                "mode": mode,
                "combine_every_quantile": combine_every_quantile,
                "threshold": threshold,
                # PMT positions, (n_pmts, 3) -- lets the notebook compute voxel-PMT distance
                # from visibility_all.positions
                "pmt_pos": torch.as_tensor(qlib.pmt_pos, dtype=torch.float32).cpu(),
                # detector fiducial-volume bounds (min, max) per axis, shape (3, 2)
                "wall_ranges": qlib.meta.ranges.cpu(),
            },
        }

        if pmt_ids:
            results["spatial_slices"] = {}
            for pid in pmt_ids:
                print(f"[eval_quantile] Building spatial slice diagnostics for PMT {pid}...")
                results["spatial_slices"][pid] = build_spatial_slice_diagnostics(
                    all_positions_tensor.numpy(),
                    all_pred_vis_tensor.numpy(),
                    all_target_vis_tensor.numpy(),
                    pmt_id=pid,
                )

        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save(results, output_file)
        print(f"[eval_quantile] Results saved to {output_file}")

    if is_distributed:
        dist.barrier()

    qlib.close()


def main():
    parser = argparse.ArgumentParser()
    default_config_path = "config/train_sirentv_81_quantile.yaml"
    parser.add_argument("--config", type=str, default=default_config_path)
    parser.add_argument("--output", type=str, default="eval_quantile_results.pt", help="output .pt file path")
    parser.add_argument(
        "--pmt-ids", type=int, nargs="+", default=None,
        help="If set, also compute spatial-slice diagnostics (high-gradient/structural vs "
             "smooth-region visibility comparison) for these PMT indices, e.g. --pmt-ids 0 40 80. "
             "One eval pass covers all of them (the diagnostics are a post-hoc step on the "
             "already-gathered results, no rerun needed per PMT).",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="Checkpoint file to evaluate. Overrides model.ckpt_file in --config -- without "
             "this, the config's own ckpt_file is used (often null, i.e. a fresh random model).",
    )
    args = parser.parse_args()

    is_distributed_env = all(k in os.environ for k in ["RANK", "WORLD_SIZE", "LOCAL_RANK"])

    if is_distributed_env:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        print(f"[main] Initialized distributed training: rank {dist.get_rank()}/{dist.get_world_size()}")
    else:
        print("[main] Running in non-distributed mode (single GPU)")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    evaluate_quantile(cfg, output_file=args.output, pmt_ids=args.pmt_ids, ckpt_file=args.ckpt)

    if is_distributed_env:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
