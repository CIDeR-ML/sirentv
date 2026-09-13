from __future__ import annotations

import argparse
import copy
import os

import torch
import torch.distributed as dist
import yaml
from tqdm import tqdm

from sirentv.data.compressed import CompressedPLib, create_compressed_dataloader
from sirentv.data.quantile import QuantilePLib
from sirentv.models import SirenTV
from sirentv.utils.comm import create_ddp_model
from sirentv.eval.utils import (
    PairwiseBiasAccumulator,
    ScalarErrorAccumulator,
    BinnedMeanAccumulator,
    gather_to_rank0,
    build_spatial_slice_diagnostics,
)


VALID_COMPARISONS = {"pred_vs_compressed", "pred_vs_raw", "compressed_vs_raw"}


@torch.no_grad()
def evaluate_pca(
    cfg: dict,
    output_file: str = "eval_pca_results.pt",
    pmt_ids: list[int] | None = None,
    ckpt_file: str | None = None,
    comparisons: list[str] | None = None,
):
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
        print(f"[eval_pca] Model device: {device}")
        print(f"[eval_pca] Distributed: {is_distributed}, world_size: {world_size}")
        if cfg.get("model", {}).get("ckpt_file"):
            print(f"[eval_pca] Loaded checkpoint: {cfg['model']['ckpt_file']}")

    # override drop_last/shuffle for eval (we want all samples, in a fixed order)
    eval_cfg = copy.deepcopy(cfg)
    if "data" in eval_cfg and "loader" in eval_cfg["data"]:
        eval_cfg["data"]["loader"]["drop_last"] = False
        eval_cfg["data"]["loader"]["shuffle"] = False
    dl = create_compressed_dataloader(eval_cfg, rank=rank, world_size=world_size)

    # Compressed PLib for CDF reconstruction (same setup as train_pca.py)
    cplib_cfg = cfg["compressed_plib"]
    cplib = CompressedPLib.load(
        cplib_cfg["filepath"],
        lazy=False,
        n_components=cplib_cfg.get("n_components"),
        device=device,
    )
    normalize_coeffs = bool(cplib_cfg.get("normalize_coeffs", False))
    if normalize_coeffs:
        cplib.compute_coeff_stats()

    n_pmts = cfg.get("data", {}).get("n_pmt", 81)
    n_quantile = cplib._u_grid.shape[-1]
    threshold = 1e-6

    # Which of the 3 pairwise comparisons to actually compute. Restricting this is useful both
    # to cut GPU memory (running all 3 at once means holding pred_time/compressed_time/raw_time
    # -- each (B, n_pmts, n_quantile) -- simultaneously) and to isolate which comparison is
    # responsible if something crashes, since pred_vs_raw/compressed_vs_raw are the only ones
    # that touch the (slow, VDS-backed) raw quantile file at all.
    raw_plib_cfg = cfg.get("raw_quantile_plib")
    if comparisons is None:
        comparisons = VALID_COMPARISONS if raw_plib_cfg is not None else {"pred_vs_compressed"}
    else:
        comparisons = set(comparisons)
        unknown = comparisons - VALID_COMPARISONS
        if unknown:
            raise ValueError(f"Unknown comparison(s) {unknown}, expected subset of {VALID_COMPARISONS}")

    do_pred_vs_compressed = "pred_vs_compressed" in comparisons
    do_pred_vs_raw = "pred_vs_raw" in comparisons
    do_compressed_vs_raw = "compressed_vs_raw" in comparisons
    compute_pca_ceiling = do_pred_vs_raw or do_compressed_vs_raw  # either needs the raw file
    if compute_pca_ceiling and raw_plib_cfg is None:
        raise RuntimeError("pred_vs_raw/compressed_vs_raw requested but raw_quantile_plib is not configured")

    if rank == 0:
        print(f"[eval_pca] Comparisons: {sorted(comparisons)}")

    if compute_pca_ceiling:
        raw_qlib = QuantilePLib.load(
            raw_plib_cfg["filepath"], lazy=True, mode=raw_plib_cfg.get("mode", "quantile"),
            combine_every_quantile=raw_plib_cfg.get("combine_every_quantile", 1),
        )
        if do_compressed_vs_raw:
            acc_compressed_vs_raw = PairwiseBiasAccumulator(n_pmts, n_quantile, threshold, device)
        if do_pred_vs_raw:
            # end-to-end error: model prediction vs. the true raw quantile function directly (not
            # just vs. the PCA-reconstructed target) -- decomposes total error into model error
            # (pred_vs_compressed) + PCA ceiling (compressed_vs_raw)
            acc_pred_vs_raw = PairwiseBiasAccumulator(n_pmts, n_quantile, threshold, device)
        if rank == 0:
            print(f"[eval_pca] Loading raw quantile file: {raw_plib_cfg['filepath']}")

    if rank == 0:
        print(f"[eval_pca] n_pmts: {n_pmts}, n_quantile: {n_quantile}, normalize_coeffs: {normalize_coeffs}")

    # pred vs. compressed (PCA-reconstructed) target, compared directly in quantile-index
    # bins (native u_grid positions) -- not interpolated onto a fixed-time CDF grid, which
    # would distort the comparison since quantile levels are equally spaced in probability,
    # not time.
    acc_pred_vs_compressed = PairwiseBiasAccumulator(n_pmts, n_quantile, threshold, device)

    # per-PMT target visibility distribution stats
    compressed_vis_sum = torch.zeros(n_pmts, device=device)
    compressed_vis_rms_sq_sum = torch.zeros(n_pmts, device=device)
    compressed_vis_count = torch.zeros(n_pmts, device=device)

    # t0 (onset) bias, restricted to visible PMTs
    t0_acc_pred_vs_compressed = ScalarErrorAccumulator(device)

    # mean adjacent-quantile-time spacing per bin (compressed-target side), for estimating the
    # local density f(Q(u)) needed by a density-corrected per-bin Poisson floor
    spacing_acc = BinnedMeanAccumulator(n_quantile - 1, device)

    log_every = 1

    all_pred_vis = []
    all_compressed_vis = []
    all_pred_t0 = []
    all_compressed_t0 = []
    all_positions = []
    all_vis_errors = []
    all_quantile_errors = []
    all_time_bias = []
    all_ceiling_time_bias = []

    for batch_idx, data in enumerate(tqdm(dl, desc=f"Rank {rank} eval", disable=(rank != 0))):
        x = data["position"].contiguous().to(device)
        target = {k: v.contiguous().to(device) for k, v in data["target"].items()}

        pred_out: dict[str, torch.Tensor] = net_module(x)

        # visibility: pred["v"] and target["v"] are in the same xformed domain
        pred_v_linear = net_module._inv_xform_vis(pred_out["v"])  # (B, n_pmts)
        compressed_v_linear = net_module._inv_xform_vis(target["v"])  # (B, n_pmts)

        # onset time: model predicts log(t0); target["t0_raw"] lives under data["meta"], not
        # data["target"], so it's not available here -- exponentiate target["t0"] instead
        # (same approach eval_quantile.py already uses correctly)
        pred_t0 = torch.exp(pred_out["t0"])  # (B, n_pmts)
        compressed_t0 = torch.exp(target["t0"])  # (B, n_pmts)

        # PCA coefficients -> raw quantile-domain reconstruction (no CDF interpolation)
        pred_coeffs = pred_out["coeffs"]
        compressed_coeffs = target["coeffs"]
        if normalize_coeffs:
            pred_coeffs = cplib.denormalize_coeffs(pred_coeffs)
            compressed_coeffs = cplib.denormalize_coeffs(compressed_coeffs)

        pred_time = cplib.to_linear_time(pred_coeffs)  # (B, n_pmts, n_quantile)
        compressed_time = cplib.to_linear_time(compressed_coeffs)  # (B, n_pmts, n_quantile)

        if do_pred_vs_compressed:
            acc_pred_vs_compressed.update(pred_v_linear, compressed_v_linear, pred_time, compressed_time)

        if compute_pca_ceiling:
            # PCA-ceiling comparison: compressed_time (PCA-reconstructed target) vs. the true
            # raw quantile function, for the same voxel/PMT pairs -- isolates how much of the
            # pred-vs-compressed bias is really "PCA can't represent this" rather than model
            # error. Reuses PairwiseBiasAccumulator's v-slot with the (uncompressed) compressed
            # visibility on both sides since only the time/quantile representation is PCA-lossy.
            voxel_id = data["meta"]["voxel_id"].numpy()  # (B,)
            raw_quantiles = raw_qlib[voxel_id]["quantiles"].to(device)  # (B, n_pmts, n_quantile)
            # zero-visibility (voxel, PMT) pairs have no quantile function -- quantiles is NaN
            # there; zero it out so NaN can't leak into the bias formula (these pairs are already
            # excluded from the accumulator via the visibility threshold)
            raw_quantiles = torch.nan_to_num(raw_quantiles, nan=0.0)
            raw_time = raw_qlib.to_linear_time(raw_quantiles)  # (B, n_pmts, n_quantile), absolute time

            # compressed_time/pred_time are t0-relative (PCA was trained on onset-subtracted
            # quantile times -- confirmed empirically: compressed_time - raw_time correlates
            # with t0 at ~-1.0 exactly), while raw_time is absolute -- add t0 back before
            # comparing against it. Use the true (target) t0 for compressed_vs_raw, since that
            # isolates PCA's own representational ceiling; use the model's own predicted t0 for
            # pred_vs_raw, since a deployed model only has its own t0 prediction available.
            compressed_time_abs = compressed_time + compressed_t0.unsqueeze(-1)
            pred_time_abs = pred_time + pred_t0.unsqueeze(-1)

            if do_compressed_vs_raw:
                acc_compressed_vs_raw.update(compressed_v_linear, compressed_v_linear, compressed_time_abs, raw_time)

                # per-(voxel, PMT) ceiling time bias, same formula/masking/reduction as
                # time_bias_per_pmt below -- lets the notebook bin the PCA-ceiling bias spatially
                # (e.g. by voxel-PMT distance) the same way it already does for the model's bias
                ceiling_mask = raw_time > threshold
                ceiling_vals = torch.where(
                    ceiling_mask,
                    2 * torch.abs(compressed_time_abs - raw_time) / (compressed_time_abs + raw_time).clamp(min=1e-10),
                    torch.zeros_like(raw_time),
                )
                ceiling_time_bias_per_pmt = ceiling_vals.sum(dim=-1) / ceiling_mask.sum(dim=-1).clamp(min=1)
                all_ceiling_time_bias.append(ceiling_time_bias_per_pmt.detach().cpu())

            if do_pred_vs_raw:
                acc_pred_vs_raw.update(pred_v_linear, compressed_v_linear, pred_time_abs, raw_time)

        # compute per-position errors for correlation check (pred vs. compressed target)
        vis_error = (pred_v_linear - compressed_v_linear).detach()  # (B, n_pmts)
        quantile_error = (pred_time - compressed_time).mean(dim=-1).detach()  # (B, n_pmts)
        # accumulator lists are held for the whole eval loop (potentially ~1.6M positions) --
        # move to CPU immediately so they don't sit on GPU for the entire run, which compounds
        # with the extra raw-quantile-shaped GPU tensor now allocated per batch when
        # compute_pca_ceiling is on
        all_vis_errors.append(vis_error.cpu())
        all_quantile_errors.append(quantile_error.cpu())

        if do_pred_vs_compressed:
            # per-(voxel, PMT) time bias, same 2*|p-t|/(p+t) formula and time-value masking as
            # PairwiseBiasAccumulator's per-tick time_bias, but reduced over the quantile-bin
            # axis instead of over voxels/PMTs -- lets the notebook bin the actual bias metric
            # (not the raw signed quantile_error, which can cancel across bins) spatially, e.g.
            # by distance from the detector wall
            time_bias_mask = compressed_time > threshold  # (B, n_pmts, n_quantile)
            time_bias_vals = torch.where(
                time_bias_mask,
                2 * torch.abs(pred_time - compressed_time) / (pred_time + compressed_time).clamp(min=1e-10),
                torch.zeros_like(pred_time),
            )
            time_bias_per_pmt = time_bias_vals.sum(dim=-1) / time_bias_mask.sum(dim=-1).clamp(min=1)  # (B, n_pmts)
            all_time_bias.append(time_bias_per_pmt.detach().cpu())

        # compressed target visibility distribution stats
        compressed_vis_mask = compressed_v_linear > threshold
        compressed_vis_masked = torch.where(compressed_vis_mask, compressed_v_linear, torch.zeros_like(compressed_v_linear))
        compressed_vis_sum += (compressed_vis_masked * compressed_vis_mask).sum(dim=0)
        compressed_vis_rms_sq_sum += torch.std((compressed_vis_masked * compressed_vis_mask), dim=0)
        compressed_vis_count += compressed_vis_mask.sum(dim=0).float()

        # t0 (onset) bias, pred vs. compressed target, restricted to visible PMTs
        t0_acc_pred_vs_compressed.update(pred_t0, compressed_t0, compressed_vis_mask)

        # adjacent-quantile-time spacing (compressed-target side), for the density-corrected
        # Poisson floor
        compressed_dt = compressed_time[..., 1:] - compressed_time[..., :-1]  # (B, n_pmts, n_quantile-1)
        spacing_acc.update(compressed_dt, compressed_vis_mask)

        # store visibility values for y=x plot
        all_pred_vis.append(pred_v_linear.detach().cpu())
        all_compressed_vis.append(compressed_v_linear.detach().cpu())
        all_pred_t0.append(pred_t0.detach().cpu())
        all_compressed_t0.append(compressed_t0.detach().cpu())
        all_positions.append(x.detach().cpu())

        # print running bias values (pred vs. compressed target) from all ranks
        if (batch_idx + 1) % log_every == 0 and is_distributed:
            local_running_vis = acc_pred_vs_compressed.overall_vis_bias_sum / acc_pred_vs_compressed.overall_vis_bias_count.clamp(min=1)
            local_running_time = acc_pred_vs_compressed.overall_time_bias_sum / acc_pred_vs_compressed.overall_time_bias_count.clamp(min=1)

            all_vis = [torch.zeros(1, device=device) for _ in range(world_size)]
            all_time = [torch.zeros(1, device=device) for _ in range(world_size)]
            dist.all_gather(all_vis, local_running_vis.unsqueeze(0))
            dist.all_gather(all_time, local_running_time.unsqueeze(0))

            if rank == 0:
                vis_strs = [f"r{i}:{v.item():.3e}" for i, v in enumerate(all_vis)]
                time_strs = [f"r{i}:{t.item():.3e}" for i, t in enumerate(all_time)]
                #print(f"  [batch {batch_idx+1}] bias_vis: [{', '.join(vis_strs)}] | bias_quantile: [{', '.join(time_strs)}]")
        elif (batch_idx + 1) % log_every == 0 and rank == 0:
            running_vis = acc_pred_vs_compressed.overall_vis_bias_sum / acc_pred_vs_compressed.overall_vis_bias_count.clamp(min=1)
            running_time = acc_pred_vs_compressed.overall_time_bias_sum / acc_pred_vs_compressed.overall_time_bias_count.clamp(min=1)
            #print(f"  [batch {batch_idx+1}] bias_vis: {running_vis.item():.4e}, bias_quantile: {running_time.item():.4e}")

    # concatenate local results
    local_results = {
        "pred_vis": torch.cat(all_pred_vis, dim=0),
        "compressed_vis": torch.cat(all_compressed_vis, dim=0),
        "pred_t0": torch.cat(all_pred_t0, dim=0),
        "compressed_t0": torch.cat(all_compressed_t0, dim=0),
        "positions": torch.cat(all_positions, dim=0),
        "vis_errors": torch.cat(all_vis_errors, dim=0),
        "quantile_errors": torch.cat(all_quantile_errors, dim=0),
    }
    if do_pred_vs_compressed:
        local_results["time_bias"] = torch.cat(all_time_bias, dim=0)
    if do_compressed_vs_raw:
        local_results["ceiling_time_bias"] = torch.cat(all_ceiling_time_bias, dim=0)

    if is_distributed:
        if do_pred_vs_compressed:
            acc_pred_vs_compressed.all_reduce()
        dist.all_reduce(compressed_vis_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(compressed_vis_rms_sq_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(compressed_vis_count, op=dist.ReduceOp.SUM)
        t0_acc_pred_vs_compressed.all_reduce()
        spacing_acc.all_reduce()
        if do_compressed_vs_raw:
            acc_compressed_vs_raw.all_reduce()
        if do_pred_vs_raw:
            acc_pred_vs_raw.all_reduce()

    gathered = gather_to_rank0(local_results, world_size, rank, device)

    if rank == 0:
        all_pred_vis_tensor = gathered["pred_vis"]
        all_compressed_vis_tensor = gathered["compressed_vis"]
        all_pred_t0_tensor = gathered["pred_t0"]
        all_compressed_t0_tensor = gathered["compressed_t0"]
        all_positions_tensor = gathered["positions"]
        all_vis_errors_tensor = gathered["vis_errors"]
        all_quantile_errors_tensor = gathered["quantile_errors"]
        if do_pred_vs_compressed:
            all_time_bias_tensor = gathered["time_bias"]
        if do_compressed_vs_raw:
            all_ceiling_time_bias_tensor = gathered["ceiling_time_bias"]

        if do_pred_vs_compressed:
            stats_pred_vs_compressed = acc_pred_vs_compressed.finalize()

        # compressed target visibility distribution stats
        compressed_vis_mean = compressed_vis_sum / compressed_vis_count.clamp(min=1)
        compressed_vis_std = torch.sqrt(compressed_vis_rms_sq_sum.clamp(min=0))

        # t0 (onset) bias stats, pred vs. compressed target
        t0_stats_pred_vs_compressed = t0_acc_pred_vs_compressed.finalize()

        # mean adjacent-quantile-time spacing per bin (compressed-target side), length
        # n_quantile-1 -- lets the notebook estimate the local density f(Q(u)) for a
        # density-corrected per-bin Poisson floor: f(Q(u_k)) ~= delta_u / quantile_time_spacing[k]
        quantile_time_spacing = spacing_acc.finalize()

        # PCA-reconstruction ceiling: compressed (PCA-reconstructed) target vs. the true raw
        # quantile function -- how much bias is inherent to the 50-component PCA compression
        # itself, independent of the model
        if do_compressed_vs_raw:
            stats_compressed_vs_raw = acc_compressed_vs_raw.finalize()
        if do_pred_vs_raw:
            # end-to-end bias: model prediction vs. the true raw quantile function directly
            stats_pred_vs_raw = acc_pred_vs_raw.finalize()

        print(f"[eval_pca] Total positions evaluated: {all_pred_vis_tensor.shape[0]}")
        if do_pred_vs_compressed:
            print(f"[eval_pca] [pred vs compressed] Overall visibility bias: {stats_pred_vs_compressed['overall']['vis_bias'].item():.6e}")
            print(f"[eval_pca] [pred vs compressed] Overall quantile-bin bias: {stats_pred_vs_compressed['overall']['time_bias'].item():.6e}")
        print(f"[eval_pca] t0 bias mean: {t0_stats_pred_vs_compressed['mean'].item():.6e}, std: {t0_stats_pred_vs_compressed['std'].item():.6e}")
        if do_compressed_vs_raw:
            print(f"[eval_pca] [compressed vs raw, PCA ceiling] Overall quantile-bin bias: {stats_compressed_vs_raw['overall']['time_bias'].item():.6e}")
        if do_pred_vs_raw:
            print(f"[eval_pca] [pred vs raw, end-to-end] Overall quantile-bin bias: {stats_pred_vs_raw['overall']['time_bias'].item():.6e}")

        # Compute visibility-quantile error correlation (pred vs. compressed target)
        vis_err_flat = all_vis_errors_tensor.flatten()
        quantile_err_flat = all_quantile_errors_tensor.flatten()
        vis_err_z = (vis_err_flat - vis_err_flat.mean()) / vis_err_flat.std()
        quantile_err_z = (quantile_err_flat - quantile_err_flat.mean()) / quantile_err_flat.std()
        correlation = torch.corrcoef(torch.stack([vis_err_z, quantile_err_z]))[0, 1]
        print(f"[eval_pca] Visibility-quantile error correlation: {correlation.item():.6f}")

        results = {
            # per-PMT target visibility distribution stats
            "compressed_target": {
                "visibility_per_pmt": {"mean": compressed_vis_mean.cpu(), "std": compressed_vis_std.cpu()},
            },
            # all visibility values for y=x plots
            "visibility_all": {
                "pred": all_pred_vis_tensor,
                "compressed_target": all_compressed_vis_tensor,
                "positions": all_positions_tensor,
            },
            "t0_all": {
                "pred": all_pred_t0_tensor,
                "target": all_compressed_t0_tensor,
            },
            "quantile_time_spacing": quantile_time_spacing,
            "error_correlation": {
                "vis_errors": all_vis_errors_tensor,
                "quantile_errors": all_quantile_errors_tensor,
                "vis_errors_z": vis_err_z,
                "quantile_errors_z": quantile_err_z,
                "correlation": correlation.cpu(),
            },
            "meta": {
                "n_pmts": n_pmts,
                # PMT positions, (n_pmts, 3) -- lets the notebook compute voxel-PMT distance
                # from visibility_all.positions, e.g. to check whether the PCA-ceiling bias (or
                # the model's own bias) concentrates at close vs. far PMTs
                "pmt_pos": torch.as_tensor(cplib.pmt_pos, dtype=torch.float32).cpu(),
                "n_quantile": n_quantile,
                "n_positions": all_pred_vis_tensor.shape[0],
                "normalize_coeffs": normalize_coeffs,
                "threshold": threshold,
                # the n_photon used to normalize target visibility (vis_raw / n_photon) --
                # needed to correctly reconstruct raw photon counts N = target_vis * n_photon,
                # since different checkpoints may have been trained/evaluated under different
                # n_photon values
                "n_photon": cplib_cfg.get("n_photon"),
                # detector fiducial-volume bounds (min, max) per axis, shape (3, 2)
                "wall_ranges": cplib.meta.ranges.cpu(),
            },
        }

        if do_pred_vs_compressed:
            # pred vs. compressed (PCA-reconstructed) target -- what the model was trained against
            results["pred_vs_compressed"] = {
                **stats_pred_vs_compressed,
                "t0_bias": t0_stats_pred_vs_compressed,
            }
            # per-(voxel, PMT) time bias (actual 2*|p-t|/(p+t) formula, not the raw signed
            # quantile_errors above) -- for binning the real bias metric spatially, e.g. by
            # distance from the detector wall
            results["error_correlation"]["time_bias"] = all_time_bias_tensor

        if do_compressed_vs_raw:
            # compressed (PCA-reconstructed) target vs. the true raw quantile function --
            # the PCA-compression ceiling, independent of the model
            results["compressed_vs_raw"] = stats_compressed_vs_raw
            # per-(voxel, PMT) version of the same comparison, same shape/semantics as
            # error_correlation.time_bias, for spatial binning (e.g. by voxel-PMT distance)
            results["error_correlation"]["ceiling_time_bias"] = all_ceiling_time_bias_tensor

        if do_pred_vs_raw:
            # end-to-end bias: model prediction vs. the true raw quantile function directly --
            # decomposes total error into model error (pred_vs_compressed) + PCA ceiling
            # (compressed_vs_raw)
            results["pred_vs_raw"] = stats_pred_vs_raw

        if pmt_ids:
            results["spatial_slices"] = {}
            for pid in pmt_ids:
                print(f"[eval_pca] Building spatial slice diagnostics for PMT {pid}...")
                results["spatial_slices"][pid] = build_spatial_slice_diagnostics(
                    all_positions_tensor.numpy(),
                    all_pred_vis_tensor.numpy(),
                    all_compressed_vis_tensor.numpy(),
                    pmt_id=pid,
                )

        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save(results, output_file)
        print(f"[eval_pca] Results saved to {output_file}")

    if is_distributed:
        dist.barrier()

    cplib.close()
    if compute_pca_ceiling:
        raw_qlib.close()


def main():
    parser = argparse.ArgumentParser()
    default_config_path = "config/default_train_cfg.yaml"
    parser.add_argument("--config", type=str, default=default_config_path)
    parser.add_argument("--output", type=str, default="eval_pca_results.pt", help="output .pt file path")
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
    parser.add_argument(
        "--compare", type=str, nargs="+", default=None, choices=sorted(VALID_COMPARISONS),
        help="Restrict to one or more of these comparisons, e.g. --compare pred_vs_compressed. "
             "Default: all three (if raw_quantile_plib is configured) or just pred_vs_compressed "
             "(if not).",
    )
    args = parser.parse_args()

    # Check if running in distributed mode (torchrun sets these env vars)
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

    evaluate_pca(cfg, output_file=args.output, pmt_ids=args.pmt_ids, ckpt_file=args.ckpt, comparisons=args.compare)

    if is_distributed_env:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
