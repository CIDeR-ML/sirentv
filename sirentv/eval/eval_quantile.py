from __future__ import annotations

import argparse
import copy
import datetime
import os

import h5py
import numpy as np
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
    build_power_spectra,
    quantile_spacing,
    sample_poisson_spacing_err,
)


@torch.no_grad()
def evaluate_quantile(
    cfg: dict,
    output_file: str = "eval_quantile_results.pt",
    pmt_ids: list[int] | None = None,
    ckpt_file: str | None = None,
    spectra: bool = True,
    spectra_components: list[int] | None = None,
    spectra_x_margin: int = 20,
    grad_cache_file: str | None = None,
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
        print(f"[eval_quantile] Model device: {device}")
        print(f"[eval_quantile] Distributed: {is_distributed}, world_size: {world_size}")
        if cfg.get("model", {}).get("ckpt_file"):
            print(f"[eval_quantile] Loaded checkpoint: {cfg['model']['ckpt_file']}")

    # override drop_last/shuffle for eval (we want all samples, in a fixed order)
    eval_cfg = copy.deepcopy(cfg)
    if "data" in eval_cfg and "loader" in eval_cfg["data"]:
        eval_cfg["data"]["loader"]["drop_last"] = False
        eval_cfg["data"]["loader"]["shuffle"] = False
    # a training run's resident shard would drop the voxels that don't divide evenly across
    # ranks; eval wants every voxel, through the ordinary sampler path
    eval_cfg.get("data", {}).pop("memory_shard", None)
    dl = create_quantile_dataloader(eval_cfg, rank=rank, world_size=world_size)

    plib_cfg = cfg.get("quantile_plib", cfg.get("photonlib", {}))
    filepath = plib_cfg["filepath"]
    mode = plib_cfg.get("mode", "quantile")
    combine_every_quantile = plib_cfg.get("combine_every_quantile", 1)
    align_t0 = bool(plib_cfg.get("align_quantiles_to_t0", False))

    # lazy=True: to_linear_time only needs the lightweight scalar attrs (mode,
    # log_quantile_C), never bulk per-voxel data.
    qlib = QuantilePLib.load(
        filepath, lazy=True, mode=mode, combine_every_quantile=combine_every_quantile,
        log_quantile_C=plib_cfg.get("log_quantile_C"),
    )

    n_pmts = cfg.get("data", {}).get("n_pmt", 81)
    n_quantile = 512 // combine_every_quantile
    threshold = 1e-6
    n_photon = float(plib_cfg.get("n_photon", 1.0))  # same source as QuantilePLibDataset._n_photon
    # u levels of the eval's quantile bins, same [::combine][:n] slicing as QuantilePLib
    with h5py.File(filepath, "r") as f:
        u_eval = torch.as_tensor(f["u_grid"][:][::combine_every_quantile][:n_quantile], dtype=torch.float32, device=device)
    # the Monte Carlo Poisson floor (sample_poisson_spacing_err) draws random numbers
    torch.manual_seed(20261005 + rank)

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
    do_spectra = bool(spectra and pmt_ids)
    if spectra_components:
        spectra_components = [int(c) for c in spectra_components]
        bad = [c for c in spectra_components if not 0 <= c < n_quantile]
        if bad:
            raise ValueError(
                f"spectra_components {bad} out of range for n_quantile={n_quantile} "
                f"(combine_every_quantile={combine_every_quantile})"
            )
    else:
        spectra_components = sorted(
            {int(round(f * (n_quantile - 1))) for f in (0.0, 0.25, 0.5, 0.75, 1.0)}
        )
    all_pred_quantiles_slice = []
    all_target_quantiles_slice = []
    all_quantiles_valid_slice = []

    if do_spectra and rank == 0:
        print(f"[eval_quantile] Power spectra enabled for PMTs {pmt_ids}, "
              f"quantile bins {spectra_components}")

    all_pred_vis = []
    all_target_vis = []
    all_pred_t0 = []
    all_target_t0 = []
    all_positions = []
    all_vis_errors = []
    all_quantile_errors = []
    all_time_bias = []
    all_time_abs_error = []
    all_time_spacing_bias = []

    for batch_idx, data in enumerate(tqdm(dl, desc=f"Rank {rank} eval", disable=(rank != 0))):
        x = data["position"].contiguous().to(device)
        target = {k: v.contiguous().to(device) for k, v in data["target"].items()}

        pred_out: dict[str, torch.Tensor] = net_module(x)
        has_v = ("v" in pred_out) and ("v" in target)
        has_t0 = "t0" in target  # t0 bias needs no v prediction (see below), just its own target
        pred_has_t0 = "t0" in pred_out  # tracked separately: whether there's a t0 PREDICTION to collect
        has_q = ("quantiles" in pred_out) and ("quantiles" in target)

        # visibility: pred["v"] and target["v"] are in the same xformed domain
        pred_v_linear = net_module._inv_xform_vis(pred_out["v"]) if has_v else None  # (B, n_pmts)
        target_v_linear = net_module._inv_xform_vis(target["v"]) if has_v else None  # (B, n_pmts)

        # onset time: model predicts log(t0), same units as target["t0"] (both log-space here)
        pred_t0 = torch.exp(pred_out["t0"]) if "t0" in pred_out else None  # (B, n_pmts)
        target_t0 = torch.exp(target["t0"]) if has_t0 else None  # (B, n_pmts)

        # zero-visibility (voxel, PMT) pairs have no quantile function -- quantiles is NaN
        # there (see quantile.py's quantiles_mask). Replace with 0 so NaN can't leak into
        # quantile_error/.mean()/.corrcoef() later.
        target_quantiles = torch.nan_to_num(target["quantiles"], nan=0.0) if has_q else None
        pred_quantiles = pred_out["quantiles"] if has_q else None

        # invert the log transform (if log_quantile mode) but stay in quantile-index space --
        # no interpolation onto a fixed time grid.
        #
        # Aligned quantiles: the target gets its own analytical_t0 back, the prediction gets
        # the model's predicted t0 -- the end-to-end absolute time a user of the model would
        # reconstruct, so t0 error shows up here too. A model with no t0 head (quantile-only
        # ablation) falls back to the target t0.
        target_t0_abs = pred_t0_abs = None
        if has_q and align_t0:
            target_t0_abs = data["meta"]["t0_raw"].to(device)
            pred_t0_abs = pred_t0 if pred_t0 is not None else target_t0_abs
        pred_time = qlib.to_linear_time(pred_quantiles, t0=pred_t0_abs) if has_q else None  # (B, n_pmts, n_quantile)
        # NaN -> 0 AFTER the linear conversion (and after t0 is added back): a log-space 0 maps to
        # 10**0 - C = 0.99 ns, which passes the target > threshold mask and leaked invisible PMTs
        # into every time metric. A linear 0 ns is excluded by that mask.
        target_time = (
            torch.nan_to_num(qlib.to_linear_time(target["quantiles"], t0=target_t0_abs), nan=0.0)
            if has_q else None
        )  # (B, n_pmts, n_quantile)

        # (voxel, PMT) pairs valid for every time metric: the same visibility cut as the
        # visibility bias (built from the raw counts, so it exists even for a quantile-only
        # model with no v target) AND the dataset's quantiles_mask (vis_raw > 0)
        if has_q:
            vis_raw = data["meta"]["v_linear"].to(device)  # (B, n_pmts) detected photon counts
            valid_vis_mask = (vis_raw / n_photon) > threshold
            valid_quantile_mask = target["quantiles_mask"].bool()
            time_pair_mask = valid_vis_mask & valid_quantile_mask  # (B, n_pmts)
            # Poisson floor of time_spacing_bias: one MC draw of the truth's own photon-count
            # fluctuation per element, |Q_hat - t| / b_k, averaged by the accumulator
            time_floor_err = sample_poisson_spacing_err(target_time, vis_raw, u_eval)
        else:
            time_pair_mask = time_floor_err = None

        acc_pred_vs_target.update(
            pred_v_linear, target_v_linear, pred_time, target_time, time_pair_mask, time_floor_err
        )

        # compute per-position errors for correlation check -- needs both predictions, so
        # there is nothing to correlate for a single-branch model missing either one
        if has_v and has_q:
            vis_error = (pred_v_linear - target_v_linear).detach()  # (B, n_pmts)
            quantile_error = (pred_time - target_time).mean(dim=-1).detach()  # (B, n_pmts)
            # accumulator lists are held for the whole eval loop (potentially ~1.6M
            # positions) -- move to CPU immediately so they don't sit on GPU the whole run
            all_vis_errors.append(vis_error.cpu())
            all_quantile_errors.append(quantile_error.cpu())

        # per-(voxel, PMT) time bias, same 2*|p-t|/(p+t) formula and time-value masking as
        # PairwiseBiasAccumulator's per-tick time_bias, but reduced over the quantile-bin axis
        # instead of over voxels/PMTs
        if has_q:
            time_bias_mask = (target_time > threshold) & time_pair_mask[..., None]  # (B, n_pmts, n_quantile)
            time_bias_vals = torch.where(
                time_bias_mask,
                2 * torch.abs(pred_time - target_time) / (pred_time + target_time).clamp(min=1e-10),
                torch.zeros_like(pred_time),
            )
            n_masked_bins = time_bias_mask.sum(dim=-1).clamp(min=1)
            time_bias_per_pmt = time_bias_vals.sum(dim=-1) / n_masked_bins  # (B, n_pmts)
            all_time_bias.append(time_bias_per_pmt.detach().cpu())
            # absolute counterpart, same mask: mean |p - t| over bins, in linear time units
            abs_err = torch.abs(pred_time - target_time)
            time_abs_err_per_pmt = torch.where(
                time_bias_mask, abs_err, torch.zeros_like(pred_time)
            ).sum(dim=-1) / n_masked_bins  # (B, n_pmts)
            all_time_abs_error.append(time_abs_err_per_pmt.detach().cpu())
            # |p - t| / b_k, b_k = local truth quantile spacing: error in units of bins. Bins with
            # b_k == 0 (degenerate truth, e.g. a single detected photon) are undefined and
            # excluded; a pair with no defined bin is NaN, not 0.
            b_k = quantile_spacing(target_time)
            spacing_mask = time_bias_mask & (b_k > 0)
            n_spacing_bins = spacing_mask.sum(dim=-1)
            time_spacing_bias_per_pmt = torch.where(
                spacing_mask, abs_err / torch.where(b_k > 0, b_k, torch.ones_like(b_k)), torch.zeros_like(pred_time)
            ).sum(dim=-1) / n_spacing_bins.clamp(min=1)  # (B, n_pmts)
            time_spacing_bias_per_pmt = torch.where(
                n_spacing_bins > 0, time_spacing_bias_per_pmt, torch.full_like(time_spacing_bias_per_pmt, float("nan"))
            )
            all_time_spacing_bias.append(time_spacing_bias_per_pmt.detach().cpu())

        # target visibility distribution stats
        if has_v:
            target_vis_mask = target_v_linear > threshold
            target_vis_masked = torch.where(target_vis_mask, target_v_linear, torch.zeros_like(target_v_linear))
            target_vis_sum += (target_vis_masked * target_vis_mask).sum(dim=0)
            target_vis_rms_sq_sum += torch.std((target_vis_masked * target_vis_mask), dim=0)
            target_vis_count += target_vis_mask.sum(dim=0).float()

        # t0 (onset) bias -- unmasked. t0 is the geometric time-of-flight from voxel to PMT;
        if has_t0:
            t0_acc_pred_vs_target.update(pred_t0, target_t0, torch.ones_like(target_t0, dtype=torch.bool))

        # adjacent-quantile-time spacing (target side), for the density-corrected Poisson
        # floor -- needs both a quantile target (to compute spacing) and a v target (to
        # restrict it to visible PMTs, since quantiles are only meaningful there, unlike t0)
        if has_q and has_v:
            target_dt = target_time[..., 1:] - target_time[..., :-1]  # (B, n_pmts, n_quantile-1)
            spacing_acc.update(target_dt, target_vis_mask)

        if do_spectra and has_q:

            sel = torch.as_tensor(pmt_ids, device=target_quantiles.device)
            comp = torch.as_tensor(spectra_components, device=target_quantiles.device)
            if "quantiles_mask" in target:
                all_quantiles_valid_slice.append(target["quantiles_mask"].detach()[:, sel].cpu())
            all_target_quantiles_slice.append(
                target_quantiles.detach()[:, sel][..., comp].cpu()
            )
            all_pred_quantiles_slice.append(
                    pred_quantiles.detach()[:, sel][..., comp].cpu()
                )

        if has_v:
            all_pred_vis.append(pred_v_linear.detach().cpu())
            all_target_vis.append(target_v_linear.detach().cpu())
        if has_t0:
            all_target_t0.append(target_t0.detach().cpu())
            if pred_has_t0:
                all_pred_t0.append(pred_t0.detach().cpu())
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
        "positions": torch.cat(all_positions, dim=0),
    }
    if has_v:
        local_results["target_vis"] = torch.cat(all_target_vis, dim=0)
        local_results["pred_vis"] = torch.cat(all_pred_vis, dim=0)
    if has_t0:
        local_results["target_t0"] = torch.cat(all_target_t0, dim=0)
        if pred_has_t0:
            local_results["pred_t0"] = torch.cat(all_pred_t0, dim=0)
    if has_q:
        local_results["time_bias"] = torch.cat(all_time_bias, dim=0)
        local_results["time_abs_error"] = torch.cat(all_time_abs_error, dim=0)
        local_results["time_spacing_bias"] = torch.cat(all_time_spacing_bias, dim=0)
        if do_spectra:
            local_results["target_quantiles_slice"] = torch.cat(all_target_quantiles_slice, dim=0)
            local_results["pred_quantiles_slice"] = torch.cat(all_pred_quantiles_slice, dim=0)
            if all_quantiles_valid_slice:
                local_results["quantiles_valid_slice"] = torch.cat(all_quantiles_valid_slice, dim=0)
    if has_v and has_q:
        local_results["vis_errors"] = torch.cat(all_vis_errors, dim=0)
        local_results["quantile_errors"] = torch.cat(all_quantile_errors, dim=0)

    if is_distributed:
        acc_pred_vs_target.all_reduce()
        dist.all_reduce(target_vis_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(target_vis_rms_sq_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(target_vis_count, op=dist.ReduceOp.SUM)
        t0_acc_pred_vs_target.all_reduce()
        spacing_acc.all_reduce()

    gathered = gather_to_rank0(local_results, world_size, rank, device)

    if rank == 0:
        all_positions_tensor = gathered["positions"]  # always present, regardless of target_keys
        all_target_vis_tensor = gathered.get("target_vis")
        all_pred_vis_tensor = gathered.get("pred_vis")
        all_target_t0_tensor = gathered.get("target_t0")
        all_pred_t0_tensor = gathered.get("pred_t0")
        all_time_bias_tensor = gathered.get("time_bias")

        stats_pred_vs_target = acc_pred_vs_target.finalize()
        t0_stats_pred_vs_target = t0_acc_pred_vs_target.finalize()
        quantile_time_spacing = spacing_acc.finalize()

        print(f"[eval_quantile] Total positions evaluated: {all_positions_tensor.shape[0]}")
        print(f"[eval_quantile] Overall visibility bias: {stats_pred_vs_target['overall']['vis_bias'].item():.6e}")
        print(f"[eval_quantile] Overall quantile-bin bias: {stats_pred_vs_target['overall']['time_bias'].item():.6e}")
        print(f"[eval_quantile] Overall quantile-bin abs error: {stats_pred_vs_target['overall']['time_abs_error'].item():.6e}")
        print(f"[eval_quantile] Overall quantile-bin spacing bias: {stats_pred_vs_target['overall']['time_spacing_bias'].item():.6e}")
        print(f"[eval_quantile] Overall quantile-bin spacing Poisson floor (MC): {stats_pred_vs_target['overall']['time_spacing_floor'].item():.6e}")
        print(f"[eval_quantile] t0 bias mean: {t0_stats_pred_vs_target['mean'].item():.6e}, std: {t0_stats_pred_vs_target['std'].item():.6e}")

        results = {
            "pred_vs_target": {
                **stats_pred_vs_target,
                "t0_bias": t0_stats_pred_vs_target,
            },
            "quantile_time_spacing": quantile_time_spacing,
            "meta": {
                "n_pmts": n_pmts,
                "n_quantile": n_quantile,
                "n_positions": all_positions_tensor.shape[0],
                "mode": mode,
                "combine_every_quantile": combine_every_quantile,
                "threshold": threshold,
                "pmt_pos": torch.as_tensor(qlib.pmt_pos, dtype=torch.float32).cpu(),
                "wall_ranges": qlib.meta.ranges.cpu(),
            },
        }

        if has_v:
            target_vis_mean = target_vis_sum / target_vis_count.clamp(min=1)
            target_vis_std = torch.sqrt(target_vis_rms_sq_sum.clamp(min=0))
            results["target"] = {
                "visibility_per_pmt": {"mean": target_vis_mean.cpu(), "std": target_vis_std.cpu()},
            }
            results["visibility_all"] = {
                "target": all_target_vis_tensor,
                "positions": all_positions_tensor,
                "pred": all_pred_vis_tensor,
            }

        if has_t0:
            results["t0_all"] = {
                "target": all_target_t0_tensor,
                **({"pred": all_pred_t0_tensor} if all_pred_t0_tensor is not None else {}),
            }

        # per-(voxel, PMT) time bias (actual 2*|p-t|/(p+t) formula, not a raw signed error)
        if has_q:
            results["error_correlation"] = {
                "time_bias": all_time_bias_tensor,
                "time_abs_error": gathered["time_abs_error"],
                "time_spacing_bias": gathered["time_spacing_bias"],
            }
            if has_v:
                all_vis_errors_tensor = gathered["vis_errors"]
                all_quantile_errors_tensor = gathered["quantile_errors"]
                vis_err_flat = all_vis_errors_tensor.flatten()
                quantile_err_flat = all_quantile_errors_tensor.flatten()
                vis_err_z = (vis_err_flat - vis_err_flat.mean()) / vis_err_flat.std()
                quantile_err_z = (quantile_err_flat - quantile_err_flat.mean()) / quantile_err_flat.std()
                correlation = torch.corrcoef(torch.stack([vis_err_z, quantile_err_z]))[0, 1]
                print(f"[eval_quantile] Visibility-quantile error correlation: {correlation.item():.6f}")
                results["error_correlation"].update({
                    "vis_errors": all_vis_errors_tensor,
                    "quantile_errors": all_quantile_errors_tensor,
                    "vis_errors_z": vis_err_z,
                    "quantile_errors_z": quantile_err_z,
                    "correlation": correlation.cpu(),
                })

        if pmt_ids and has_v:
            results["spatial_slices"] = {}
            for pid in pmt_ids:
                print(f"[eval_quantile] Building spatial slice diagnostics for PMT {pid}...")
                results["spatial_slices"][pid] = build_spatial_slice_diagnostics(
                    all_positions_tensor.numpy(),
                    all_pred_vis_tensor.numpy(),
                    all_target_vis_tensor.numpy(),
                    pmt_id=pid,
                )
        elif pmt_ids and rank == 0:
            print("[eval_quantile] Skipping spatial-slice diagnostics: this model has no "
                  "v prediction to compare against.")

        if do_spectra:
            positions_np = all_positions_tensor.numpy()

            target_vis_np = all_target_vis_tensor.numpy() if has_v else None
            pred_quantiles_np = gathered["pred_quantiles_slice"].numpy() if has_q else None
            target_quantiles_np = gathered["target_quantiles_slice"].numpy() if has_q else None
            # quantiles_mask (vis_raw > 0) is read whenever "quantiles" is in target_keys,
            # independent of whether "v" is also exposed as its own target key
            quantiles_valid_np = gathered["quantiles_valid_slice"].numpy() if "quantiles_valid_slice" in gathered else None

            # Cached gradient targets are per-(voxel, PMT) and indexed by voxel_id over the
            # FULL LUT, while this eval may have run on a subset -- so it is only usable when
            # the row counts line up.
            grad_targets = None
            if grad_cache_file:
                with h5py.File(grad_cache_file, "r") as gf:
                    if gf["v"].shape[0] == positions_np.shape[0]:
                        grad_targets = {
                            "grad_v": gf["v"][:],
                            "grad_quantiles": gf["quantiles"][:],
                        }
                    else:
                        print(f"[eval_quantile] grad cache has {gf['v'].shape[0]} rows but eval "
                              f"covered {positions_np.shape[0]} -- skipping gradient spectra")

            spectra_variants = {
                "power_spectra": {},
                f"power_spectra_x_far{spectra_x_margin}": {
                    "x_margin_voxels": spectra_x_margin, "x_margin_mode": "either",
                },
                f"power_spectra_x_far_pmtwall{spectra_x_margin}": {
                    "x_margin_voxels": spectra_x_margin, "x_margin_mode": "pmt_wall",
                },
                "power_spectra_zerofill": {"inpaint": False},
            }
            for variant_key in spectra_variants:
                results[variant_key] = {}
            for i, pid in enumerate(pmt_ids):
                print(f"[eval_quantile] Building power spectra for PMT {pid}...")
                if has_v:
                    valid = target_vis_np[:, pid] > 0
                elif quantiles_valid_np is not None:
                    valid = quantiles_valid_np[:, i] > 0
                else:
                    valid = np.ones(positions_np.shape[0], dtype=bool)
                fields = {}
                if has_v:
                    fields["v"] = {"target": target_vis_np[:, pid], "pred": all_pred_vis_tensor.numpy()[:, pid]}
                if has_t0:
                    fields["t0"] = {"target": all_target_t0_tensor.numpy()[:, pid]}
                    if all_pred_t0_tensor is not None:
                        fields["t0"]["pred"] = all_pred_t0_tensor.numpy()[:, pid]
                if has_q:
                    fields["quantiles"] = {"target": target_quantiles_np[:, i, :], "pred": pred_quantiles_np[:, i, :]}
                if grad_targets is not None:
                    fields["grad_v"] = {"target": grad_targets["grad_v"][:, pid]}
                    fields["grad_quantiles"] = {"target": grad_targets["grad_quantiles"][:, pid]}

                for variant_key, variant_kwargs in spectra_variants.items():
                    spec = build_power_spectra(positions_np, fields, valid, **variant_kwargs)
                    spec["_components"] = list(spectra_components)
                    results[variant_key][pid] = spec

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
    parser.add_argument(
        "--spectra", action="store_true", default=True,
        help="Also compute (kx, ky, kz) spatial-frequency power spectra of v/t0/quantiles, "
             "truth vs prediction, for each --pmt-ids PMT. On by default; needs --pmt-ids "
             "to actually produce anything (silently a no-op without it). Use --no-spectra "
             "to skip (e.g. to avoid the extra compute when you don't need it).",
    )
    parser.add_argument("--no-spectra", action="store_false", dest="spectra")
    parser.add_argument(
        "--spectra-components", type=int, nargs="+", default=None,
        help="Quantile bin indices to include in the quantile spectra (default: five evenly "
             "spaced levels u ~ 0, 0.25, 0.5, 0.75, 1). Only used with --spectra.",
    )
    parser.add_argument(
        "--spectra-x-margin", type=int, default=20,
        help="Voxel margin for the x-boundary-excluded spectra variant "
             "(power_spectra_x_far<N>): only voxels more than this many cells from EITHER "
             "x boundary are included, i.e. the near-PMT-wall region is excluded. Only "
             "used with --spectra.",
    )
    parser.add_argument(
        "--grad-cache", type=str, default=None,
        help="Cached grad_frob targets (HDF5 with 'v' and 'quantiles'). When its row count "
             "matches the evaluated set, target-side grad_v/grad_quantiles spectra are added. "
             "Prediction-side gradient spectra are not available under no_grad eval. Defaults "
             "to train.grad_target_cache_file from the config.",
    )
    args = parser.parse_args()

    is_distributed_env = all(k in os.environ for k in ["RANK", "WORLD_SIZE", "LOCAL_RANK"])

    if is_distributed_env:
        # see eval.py's own init_process_group call for why this needs a non-default timeout
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
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

    evaluate_quantile(
        cfg,
        output_file=args.output,
        pmt_ids=args.pmt_ids,
        ckpt_file=args.ckpt,
        spectra=args.spectra,
        spectra_components=args.spectra_components,
        spectra_x_margin=args.spectra_x_margin,
        grad_cache_file=args.grad_cache or cfg.get("train", {}).get("grad_target_cache_file"),
    )

    if is_distributed_env:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
