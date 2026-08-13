from __future__ import annotations

import argparse
import copy
import os
from typing import Literal

import torch
import torch.distributed as dist
import yaml

from sirentv.data.builder import create_dataloader
from sirentv.models import SirenTV
from sirentv.utils.comm import create_ddp_model
from sirentv.analysis import bias as compute_bias
from tqdm import tqdm


@torch.no_grad()
def evaluate(cfg: dict, output_file: str = "eval_results.pt"):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    is_distributed = world_size > 1

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    net = SirenTV(cfg).to(device)
    if is_distributed:
        net = create_ddp_model(net, device_ids=[local_rank])
    net.eval()

    net_module = net.module if is_distributed else net
    mode: Literal["pdf", "cdf"] = net_module.mode

    if rank == 0:
        print(f"[eval] Model device: {device}")
        print(f"[eval] Distributed: {is_distributed}, world_size: {world_size}")
        if cfg.get("model", {}).get("ckpt_file"):
            print(f"[eval] Loaded checkpoint: {cfg['model']['ckpt_file']}")

    # override drop_last to False for eval (we want all samples)
    eval_cfg = copy.deepcopy(cfg)
    if "data" in eval_cfg and "loader" in eval_cfg["data"]:
        eval_cfg["data"]["loader"]["drop_last"] = False
        eval_cfg["data"]["loader"]["shuffle"] = False
    dl = create_dataloader(eval_cfg, rank=rank, world_size=world_size)
    inv_xform_vis = net_module._inv_xform_vis

    n_pmts = cfg.get("data", {}).get("n_pmt", 81)
    n_ticks = cfg.get("model", {}).get("network", {}).get("out_features", [1, 1001])[1] - 1
    threshold = cfg.get("eval", {}).get("threshold", 1e-6)
    tick_size = cfg.get("photonlib", {}).get("time_tick_size", 0.1)  # ns

    if rank == 0:
        print(f"[eval] Mode: {mode}, tick_size: {tick_size} ns")

    # accumulators for per-PMT visibility bias (using analysis.bias formula)
    target_vis_sum = torch.zeros(n_pmts, device=device)
    target_vis_sq_sum = torch.zeros(n_pmts, device=device)
    vis_bias_sum = torch.zeros(n_pmts, device=device)
    vis_bias_sq_sum = torch.zeros(n_pmts, device=device)
    vis_count = torch.zeros(n_pmts, device=device)

    # accumulators for per-tick time bias
    time_bias_sum = torch.zeros(n_ticks, device=device)
    time_bias_sq_sum = torch.zeros(n_ticks, device=device)
    time_count = torch.zeros(n_ticks, device=device)

    # accumulators for overall bias (matching training exactly)
    overall_vis_bias_sum = torch.tensor(0.0, device=device)
    overall_vis_bias_count = torch.tensor(0, device=device)
    overall_time_bias_sum = torch.tensor(0.0, device=device)
    overall_time_bias_count = torch.tensor(0, device=device)

    # for running average display
    log_every = 1

    # storage for all visibility values
    all_pred_vis = []
    all_true_vis = []
    all_positions = []
    all_vis_errors = []
    all_pdf_errors = []

    for batch_idx, data in enumerate(tqdm(dl, desc=f"Rank {rank} eval", disable=(rank != 0))):
        x = data["position"].contiguous().to(device)
        meta = {
            key: value.contiguous().to(device)
            for key, value in data.get("meta", {}).items()
        }
        if "t_linear" not in meta or "v_linear" not in meta:
            raise KeyError(
                "Evaluation requires standardized dataset metadata keys "
                "'t_linear' and 'v_linear'"
            )
        target_t_for_bias = meta["t_linear"]
        target_v_linear = meta["v_linear"]

        pred_out: dict[str, torch.Tensor] = net_module(x)

        # prepare pred and target dicts
        pred_v_linear = inv_xform_vis(pred_out["v"])  # (B, n_pmts)

        # PLibDataset stores CDF metadata in CDF mode and PDF metadata in PDF mode.
        pred_t_for_bias = pred_out["t"]

        target = {
            "v_linear": target_v_linear,
            "t_linear": target_t_for_bias,
        }
        pred = {
            "v_linear": pred_v_linear,
            "t_linear": pred_t_for_bias,
        }

        # compute overall bias using the actual analysis.bias function (for sanity check)
        # weight by number of masked elements, not total elements
        vis_masked_count = (target_v_linear > threshold).sum()
        time_masked_count = (target_t_for_bias > threshold).sum()

        batch_vis_bias = compute_bias(target, pred, key="v_linear", threshold=threshold)
        batch_time_bias = compute_bias(target, pred, key="t_linear", threshold=threshold)
        overall_vis_bias_sum += batch_vis_bias * vis_masked_count
        overall_vis_bias_count += vis_masked_count
        overall_time_bias_sum += batch_time_bias * time_masked_count
        overall_time_bias_count += time_masked_count

        # compute per-position errors for correlation check
        vis_error = (pred_v_linear - target_v_linear).detach()  # (B, n_pmts)
        # mean PDF/CDF error across ticks per PMT
        pdf_error = (pred_t_for_bias - target_t_for_bias).mean(dim=-1).detach()  # (B, n_pmts)
        all_vis_errors.append(vis_error)
        all_pdf_errors.append(pdf_error)


        # per-PMT visibility bias: 2 * |p - t| / (p + t) for each PMT
        vis_mask = target_v_linear > threshold  # (B, n_pmts)
        p = pred_v_linear
        t = target_v_linear
        vis_bias_vals = torch.where(
            vis_mask,
            2 * torch.abs(p - t) / (p + t).clamp(min=1e-10),
            torch.zeros_like(p)
        )
        target_vis_masked = torch.where(
            vis_mask,
            t,
            torch.zeros_like(t)
        )

        target_vis_sum += (target_vis_masked * vis_mask).sum(dim=0)
        target_vis_sq_sum += ((target_vis_masked ** 2) * vis_mask).sum(dim=0)
        vis_bias_sum += (vis_bias_vals * vis_mask).sum(dim=0)
        vis_bias_sq_sum += ((vis_bias_vals ** 2) * vis_mask).sum(dim=0)
        vis_count += vis_mask.sum(dim=0).float()

        # per-tick time bias: 2 * |p - t| / (p + t) averaged across PMTs per tick
        actual_n_ticks = min(n_ticks, pred_t_for_bias.shape[-1], target_t_for_bias.shape[-1])
        pred_t = pred_t_for_bias[..., :actual_n_ticks]
        target_t = target_t_for_bias[..., :actual_n_ticks]

        time_mask = target_t > threshold  # (B, n_pmts, n_ticks)
        time_bias_vals = torch.where(
            time_mask,
            2 * torch.abs(pred_t - target_t) / (pred_t + target_t).clamp(min=1e-10),
            torch.zeros_like(pred_t)
        )
        # mean across PMTs, then accumulate
        time_bias_per_tick = time_bias_vals.sum(dim=1) / time_mask.sum(dim=1).clamp(min=1)  # (B, n_ticks)
        time_mask_any = time_mask.any(dim=1)  # (B, n_ticks)

        time_bias_sum[:actual_n_ticks] += (time_bias_per_tick * time_mask_any).sum(dim=0)
        time_bias_sq_sum[:actual_n_ticks] += ((time_bias_per_tick ** 2) * time_mask_any).sum(dim=0)
        time_count[:actual_n_ticks] += time_mask_any.sum(dim=0).float()

        # store visibility values for y=x plot
        all_pred_vis.append(pred_v_linear.detach())
        all_true_vis.append(target_v_linear.detach())
        all_positions.append(x.detach())

        # print running bias values from all ranks
        if (batch_idx + 1) % log_every == 0 and is_distributed:
            # gather running values from all ranks
            local_running_vis = overall_vis_bias_sum / overall_vis_bias_count.clamp(min=1)
            local_running_time = overall_time_bias_sum / overall_time_bias_count.clamp(min=1)

            all_vis = [torch.zeros(1, device=device) for _ in range(world_size)]
            all_time = [torch.zeros(1, device=device) for _ in range(world_size)]
            dist.all_gather(all_vis, local_running_vis.unsqueeze(0))
            dist.all_gather(all_time, local_running_time.unsqueeze(0))

            if rank == 0:
                vis_strs = [f"r{i}:{v.item():.3e}" for i, v in enumerate(all_vis)]
                time_strs = [f"r{i}:{t.item():.3e}" for i, t in enumerate(all_time)]
                print(f"  [batch {batch_idx+1}] bias_vis: [{', '.join(vis_strs)}] | bias_time: [{', '.join(time_strs)}]")
        elif (batch_idx + 1) % log_every == 0 and rank == 0:
            # non-distributed case
            running_vis = overall_vis_bias_sum / overall_vis_bias_count.clamp(min=1)
            running_time = overall_time_bias_sum / overall_time_bias_count.clamp(min=1)
            print(f"  [batch {batch_idx+1}] bias_vis: {running_vis.item():.4e}, bias_time: {running_time.item():.4e}")

    # concatenate local results
    local_pred_vis = torch.cat(all_pred_vis, dim=0)
    local_true_vis = torch.cat(all_true_vis, dim=0)
    local_positions = torch.cat(all_positions, dim=0)
    local_vis_errors = torch.cat(all_vis_errors, dim=0)
    local_pdf_errors = torch.cat(all_pdf_errors, dim=0)

    if is_distributed:
        # reduce all statistics across ranks
        dist.all_reduce(vis_bias_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(vis_bias_sq_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(vis_count, op=dist.ReduceOp.SUM)

        dist.all_reduce(target_vis_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(target_vis_sq_sum, op=dist.ReduceOp.SUM)

        dist.all_reduce(time_bias_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(time_bias_sq_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(time_count, op=dist.ReduceOp.SUM)

        dist.all_reduce(overall_vis_bias_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(overall_vis_bias_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(overall_time_bias_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(overall_time_bias_count, op=dist.ReduceOp.SUM)

        # gather all visibility values to rank 0
        local_sizes = torch.tensor([local_pred_vis.shape[0]], device=device)
        all_sizes = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_sizes)
        all_sizes = [int(s.item()) for s in all_sizes]

        if rank == 0:
            gathered_pred_vis = [torch.zeros(sz, n_pmts, device=device) for sz in all_sizes]
            gathered_true_vis = [torch.zeros(sz, n_pmts, device=device) for sz in all_sizes]
            gathered_positions = [torch.zeros(sz, 3, device=device) for sz in all_sizes]

            gathered_vis_errors = [torch.zeros(sz, n_pmts, device=device) for sz in all_sizes]
            gathered_pdf_errors = [torch.zeros(sz, n_pmts, device=device) for sz in all_sizes]
        else:
            gathered_pred_vis = None
            gathered_true_vis = None
            gathered_positions = None

            gathered_vis_errors = None
            gathered_pdf_errors = None

        dist.gather(local_pred_vis, gathered_pred_vis if rank == 0 else None, dst=0)
        dist.gather(local_true_vis, gathered_true_vis if rank == 0 else None, dst=0)
        dist.gather(local_positions, gathered_positions if rank == 0 else None, dst=0)
        dist.gather(local_vis_errors, gathered_vis_errors if rank == 0 else None, dst=0)
        dist.gather(local_pdf_errors, gathered_pdf_errors if rank == 0 else None, dst=0)

        if rank == 0:
            all_pred_vis_tensor = torch.cat(gathered_pred_vis, dim=0).cpu()
            all_true_vis_tensor = torch.cat(gathered_true_vis, dim=0).cpu()
            all_positions_tensor = torch.cat(gathered_positions, dim=0).cpu()
            all_vis_errors_tensor = torch.cat(gathered_vis_errors, dim=0).cpu()
            all_pdf_errors_tensor = torch.cat(gathered_pdf_errors, dim=0).cpu()
    else:
        all_pred_vis_tensor = local_pred_vis.cpu()
        all_true_vis_tensor = local_true_vis.cpu()
        all_positions_tensor = local_positions.cpu()
        all_vis_errors_tensor = local_vis_errors.cpu()
        all_pdf_errors_tensor = local_pdf_errors.cpu()

    if rank == 0:
        # overall bias (matches training logger)
        overall_vis_bias = overall_vis_bias_sum / overall_vis_bias_count.clamp(min=1)
        overall_time_bias = overall_time_bias_sum / overall_time_bias_count.clamp(min=1)

        # per-PMT visibility bias stats
        vis_bias_mean = vis_bias_sum / vis_count.clamp(min=1)
        vis_bias_var = (vis_bias_sq_sum / vis_count.clamp(min=1)) - vis_bias_mean ** 2
        vis_bias_std = torch.sqrt(vis_bias_var.clamp(min=0))
        vis_bias_sem = vis_bias_std / torch.sqrt(vis_count.clamp(min=1))

        # per-PMT target visibility stats
        target_vis_mean = target_vis_sum / vis_count.clamp(min=1)
        target_vis_var = (target_vis_sq_sum / vis_count.clamp(min=1)) - target_vis_mean ** 2
        target_vis_std = torch.sqrt(target_vis_var.clamp(min=0))

        # per-tick time bias stats
        time_bias_mean = time_bias_sum / time_count.clamp(min=1)
        time_bias_var = (time_bias_sq_sum / time_count.clamp(min=1)) - time_bias_mean ** 2
        time_bias_std = torch.sqrt(time_bias_var.clamp(min=0))
        time_bias_sem = time_bias_std / torch.sqrt(time_count.clamp(min=1))

        print(f"[eval] Total positions evaluated: {all_pred_vis_tensor.shape[0]}")
        print(f"[eval] Overall visibility bias (analysis.bias): {overall_vis_bias.item():.6e}")
        print(f"[eval] Overall time bias (analysis.bias): {overall_time_bias.item():.6e}")
        print(f"[eval] Per-PMT visibility bias mean: {vis_bias_mean.mean().item():.6e}")
        print(f"[eval] Per-tick time bias mean: {time_bias_mean.mean().item():.6e}")
        # Compute visibility-PDF error correlation
        # Flatten across positions and PMTs: (N_positions * N_pmts,)
        vis_err_flat = all_vis_errors_tensor.flatten()
        pdf_err_flat = all_pdf_errors_tensor.flatten()

        # Z-score normalization
        vis_err_z = (vis_err_flat - vis_err_flat.mean()) / vis_err_flat.std()
        pdf_err_z = (pdf_err_flat - pdf_err_flat.mean()) / pdf_err_flat.std()

        # Compute correlation coefficient
        correlation = torch.corrcoef(torch.stack([vis_err_z, pdf_err_z]))[0, 1]
        print(f"[eval] Visibility-PDF error correlation: {correlation.item():.6f}")

        results = {
            # overall bias (matches training)
            "overall": {
                "vis_bias": overall_vis_bias.cpu(),
                "time_bias": overall_time_bias.cpu(),
            },
            # per-PMT visibility bias (81 values)
            "visibility_bias": {
                "mean": vis_bias_mean.cpu(),
                "std": vis_bias_std.cpu(),
                "sem": vis_bias_sem.cpu(),
                "count": vis_count.cpu(),
            },
            # per-PMT visibility stats (81 values)
            "visibility_per_pmt":{
                "mean": target_vis_mean.cpu(),
                "std": target_vis_std.cpu(),
            },
            # per-tick time bias (1000 values)
            "time_bias": {
                "mean": time_bias_mean.cpu(),
                "std": time_bias_std.cpu(),
                "sem": time_bias_sem.cpu(),
                "count": time_count.cpu(),
            },
            # all visibility values for y=x plot
            "visibility_all": {
                "pred": all_pred_vis_tensor,
                "true": all_true_vis_tensor,
                "positions": all_positions_tensor,
            },
            "error_correlation": {
                "vis_errors": all_vis_errors_tensor,
                "pdf_errors": all_pdf_errors_tensor,
                "vis_errors_z": vis_err_z,
                "cdf_err_z": pdf_err_z,
                "correlation": correlation.cpu(),
            },
            "meta": {
                "n_pmts": n_pmts,
                "n_ticks": n_ticks,
                "n_positions": all_pred_vis_tensor.shape[0],
                "mode": mode,
                "threshold": threshold,
            },
        }
        torch.save(results, output_file)
        print(f"[eval] Results saved to {output_file}")

    if is_distributed:
        dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default="eval_results.pt", help="output .pt file path")
    args = parser.parse_args()

    # Check if running in distributed mode (torchrun sets these env vars)
    is_distributed_env = all(k in os.environ for k in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK'])

    if is_distributed_env:
        # Running with torchrun - initialize distributed
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        print(f"[main] Initialized distributed training: rank {dist.get_rank()}/{dist.get_world_size()}")
    else:
        # Running standalone (python train.py) - single GPU
        print("[main] Running in non-distributed mode (single GPU)")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg, output_file=args.output)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
