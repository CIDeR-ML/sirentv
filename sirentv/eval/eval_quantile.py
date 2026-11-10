from __future__ import annotations

import h5py
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from sirentv.data.builder import create_dataloader
from sirentv.data.quantile import QuantilePLib
from sirentv.eval.utils import (
    BinnedMeanAccumulator,
    PairwiseBiasAccumulator,
    ScalarErrorAccumulator,
    VisStatsAccumulator,
    add_power_spectra,
    build_spatial_slice_diagnostics,
    error_correlation,
    eval_loader_cfg,
    gather_to_rank0,
    load_grad_targets,
    masked_bin_bias,
    quantile_spacing,
    sample_poisson_spacing_err,
    save_results,
    setup_eval_model,
)

TAG = "eval_quantile"


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
    """Evaluate a quantile model against its LUT, compared in native quantile bins (u_grid),
    not interpolated onto a time grid. Any of v / t0 / quantiles may be missing from the model."""
    rank, world_size, device, net_module = setup_eval_model(cfg, ckpt_file, TAG)
    is_distributed = world_size > 1

    eval_cfg = eval_loader_cfg(cfg)
    # a training shard drops the voxels that don't divide evenly across ranks
    eval_cfg.get("data", {}).pop("memory_shard", None)
    dl = create_dataloader(eval_cfg, rank=rank, world_size=world_size)

    plib_cfg = cfg.get("quantile_plib", cfg.get("photonlib", {}))
    filepath = plib_cfg["filepath"]
    mode = plib_cfg.get("mode", "quantile")
    combine_every_quantile = plib_cfg.get("combine_every_quantile", 1)
    align_t0 = bool(plib_cfg.get("align_quantiles_to_t0", False))

    # lazy: to_linear_time only needs scalar attributes
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
        print(f"[{TAG}] n_pmts: {n_pmts}, n_quantile: {n_quantile}, mode: {mode}, combine_every_quantile: {combine_every_quantile}")

    acc_pred_vs_target = PairwiseBiasAccumulator(n_pmts, n_quantile, threshold, device)
    vis_stats = VisStatsAccumulator(n_pmts, threshold, device)
    t0_acc_pred_vs_target = ScalarErrorAccumulator(device)
    # target-side adjacent-quantile spacing, for a density-corrected Poisson floor
    spacing_acc = BinnedMeanAccumulator(n_quantile - 1, device)

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
    if do_spectra and rank == 0:
        print(f"[{TAG}] Power spectra enabled for PMTs {pmt_ids}, "
              f"quantile bins {spectra_components}")

    # per-batch results, kept on CPU for the whole loop
    collected = {k: [] for k in (
        "positions", "target_vis", "pred_vis", "target_t0", "pred_t0",
        "time_bias", "time_abs_error", "time_spacing_bias",
        "target_quantiles_slice", "pred_quantiles_slice", "quantiles_valid_slice",
        "vis_errors", "quantile_errors",
    )}

    for data in tqdm(dl, desc=f"Rank {rank} eval", disable=(rank != 0)):
        x = data["position"].contiguous().to(device)
        target = {k: v.contiguous().to(device) for k, v in data["target"].items()}

        pred_out: dict[str, torch.Tensor] = net_module(x)
        has_v = ("v" in pred_out) and ("v" in target)
        has_t0 = "t0" in target  # t0 bias needs only its own target
        pred_has_t0 = "t0" in pred_out
        has_q = ("quantiles" in pred_out) and ("quantiles" in target)

        pred_v_linear = net_module._inv_xform_vis(pred_out["v"]) if has_v else None  # (B, n_pmts)
        target_v_linear = net_module._inv_xform_vis(target["v"]) if has_v else None

        # both sides are log(t0)
        pred_t0 = torch.exp(pred_out["t0"]) if "t0" in pred_out else None  # (B, n_pmts)
        target_t0 = torch.exp(target["t0"]) if has_t0 else None

        # zero-visibility pairs have NaN quantiles (see quantiles_mask)
        target_quantiles = torch.nan_to_num(target["quantiles"], nan=0.0) if has_q else None
        pred_quantiles = pred_out["quantiles"] if has_q else None

        # Aligned quantiles get their t0 back: the target its true t0, the prediction its own
        # predicted t0 (end-to-end), falling back to the target's for a model with no t0 head.
        target_t0_abs = pred_t0_abs = None
        if has_q and align_t0:
            target_t0_abs = data["meta"]["t0_raw"].to(device)
            pred_t0_abs = pred_t0 if pred_t0 is not None else target_t0_abs
        pred_time = qlib.to_linear_time(pred_quantiles, t0=pred_t0_abs) if has_q else None  # (B, n_pmts, n_quantile)
        # NaN -> 0 AFTER the linear conversion: a log-space 0 maps to 0.99 ns, which would pass
        # the target > threshold mask and leak invisible PMTs into every time metric.
        target_time = (
            torch.nan_to_num(qlib.to_linear_time(target["quantiles"], t0=target_t0_abs), nan=0.0)
            if has_q else None
        )

        # pairs valid for every time metric: the visibility cut (from raw counts, so it exists
        # without a v target) AND quantiles_mask (vis_raw > 0)
        if has_q:
            vis_raw = data["meta"]["v_linear"].to(device)  # (B, n_pmts) detected photon counts
            valid_vis_mask = (vis_raw / n_photon) > threshold
            valid_quantile_mask = target["quantiles_mask"].bool()
            time_pair_mask = valid_vis_mask & valid_quantile_mask
            time_floor_err = sample_poisson_spacing_err(target_time, vis_raw, u_eval)
        else:
            time_pair_mask = time_floor_err = None

        acc_pred_vs_target.update(
            pred_v_linear, target_v_linear, pred_time, target_time, time_pair_mask, time_floor_err
        )

        if has_v and has_q:
            collected["vis_errors"].append((pred_v_linear - target_v_linear).detach().cpu())
            collected["quantile_errors"].append((pred_time - target_time).mean(dim=-1).detach().cpu())

        # per-(voxel, PMT) time metrics, reduced over the quantile bins
        if has_q:
            time_bias_mask = (target_time > threshold) & time_pair_mask[..., None]  # (B, n_pmts, n_quantile)
            n_masked_bins = time_bias_mask.sum(dim=-1).clamp(min=1)
            collected["time_bias"].append(masked_bin_bias(pred_time, target_time, time_bias_mask).detach().cpu())
            abs_err = torch.abs(pred_time - target_time)
            time_abs_err_per_pmt = torch.where(
                time_bias_mask, abs_err, torch.zeros_like(pred_time)
            ).sum(dim=-1) / n_masked_bins
            collected["time_abs_error"].append(time_abs_err_per_pmt.detach().cpu())
            # error in units of the local truth spacing; b_k == 0 bins are undefined and excluded,
            # a pair with no defined bin is NaN
            b_k = quantile_spacing(target_time)
            spacing_mask = time_bias_mask & (b_k > 0)
            n_spacing_bins = spacing_mask.sum(dim=-1)
            time_spacing_bias_per_pmt = torch.where(
                spacing_mask, abs_err / torch.where(b_k > 0, b_k, torch.ones_like(b_k)), torch.zeros_like(pred_time)
            ).sum(dim=-1) / n_spacing_bins.clamp(min=1)
            time_spacing_bias_per_pmt = torch.where(
                n_spacing_bins > 0, time_spacing_bias_per_pmt, torch.full_like(time_spacing_bias_per_pmt, float("nan"))
            )
            collected["time_spacing_bias"].append(time_spacing_bias_per_pmt.detach().cpu())

        if has_v:
            target_vis_mask = vis_stats.update(target_v_linear)

        # t0 is the geometric time of flight, so its error is not masked
        if has_t0:
            t0_acc_pred_vs_target.update(pred_t0, target_t0, torch.ones_like(target_t0, dtype=torch.bool))

        if has_q and has_v:
            target_dt = target_time[..., 1:] - target_time[..., :-1]  # (B, n_pmts, n_quantile-1)
            spacing_acc.update(target_dt, target_vis_mask)

        if do_spectra and has_q:
            sel = torch.as_tensor(pmt_ids, device=target_quantiles.device)
            comp = torch.as_tensor(spectra_components, device=target_quantiles.device)
            if "quantiles_mask" in target:
                collected["quantiles_valid_slice"].append(target["quantiles_mask"].detach()[:, sel].cpu())
            collected["target_quantiles_slice"].append(target_quantiles.detach()[:, sel][..., comp].cpu())
            collected["pred_quantiles_slice"].append(pred_quantiles.detach()[:, sel][..., comp].cpu())

        if has_v:
            collected["pred_vis"].append(pred_v_linear.detach().cpu())
            collected["target_vis"].append(target_v_linear.detach().cpu())
        if has_t0:
            collected["target_t0"].append(target_t0.detach().cpu())
            if pred_has_t0:
                collected["pred_t0"].append(pred_t0.detach().cpu())
        collected["positions"].append(x.detach().cpu())

    local_results = {k: torch.cat(v, dim=0) for k, v in collected.items() if v}

    if is_distributed:
        acc_pred_vs_target.all_reduce()
        vis_stats.all_reduce()
        t0_acc_pred_vs_target.all_reduce()
        spacing_acc.all_reduce()

    gathered = gather_to_rank0(local_results, world_size, rank, device)

    if rank == 0:
        all_positions_tensor = gathered["positions"]
        all_target_vis_tensor = gathered.get("target_vis")
        all_pred_vis_tensor = gathered.get("pred_vis")
        all_target_t0_tensor = gathered.get("target_t0")
        all_pred_t0_tensor = gathered.get("pred_t0")

        stats_pred_vs_target = acc_pred_vs_target.finalize()
        t0_stats_pred_vs_target = t0_acc_pred_vs_target.finalize()
        overall = stats_pred_vs_target["overall"]

        print(f"[{TAG}] Total positions evaluated: {all_positions_tensor.shape[0]}")
        print(f"[{TAG}] Overall visibility bias: {overall['vis_bias'].item():.6e}")
        print(f"[{TAG}] Overall quantile-bin bias: {overall['time_bias'].item():.6e}")
        print(f"[{TAG}] Overall quantile-bin abs error: {overall['time_abs_error'].item():.6e}")
        print(f"[{TAG}] Overall quantile-bin spacing bias: {overall['time_spacing_bias'].item():.6e}")
        print(f"[{TAG}] Overall quantile-bin spacing Poisson floor (MC): {overall['time_spacing_floor'].item():.6e}")
        print(f"[{TAG}] t0 bias mean: {t0_stats_pred_vs_target['mean'].item():.6e}, std: {t0_stats_pred_vs_target['std'].item():.6e}")

        results = {
            "pred_vs_target": {**stats_pred_vs_target, "t0_bias": t0_stats_pred_vs_target},
            "quantile_time_spacing": spacing_acc.finalize(),
            "meta": {
                "n_pmts": n_pmts,
                "n_quantile": n_quantile,
                "n_positions": all_positions_tensor.shape[0],
                "mode": mode,
                "combine_every_quantile": combine_every_quantile,
                "threshold": threshold,
                "pmt_pos": torch.as_tensor(qlib.pmt_pos, dtype=torch.float32).cpu(),
                "wall_ranges": qlib.meta.ranges.cpu(),
                "x_min": cfg.get("data", {}).get("x_min"),  # [mm]; None = every LUT voxel
            },
        }

        if has_v:
            results["target"] = {"visibility_per_pmt": vis_stats.finalize()}
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

        if has_q:
            results["error_correlation"] = {
                "time_bias": gathered.get("time_bias"),
                "time_abs_error": gathered["time_abs_error"],
                "time_spacing_bias": gathered["time_spacing_bias"],
            }
            if has_v:
                corr = error_correlation(gathered["vis_errors"], gathered["quantile_errors"])
                print(f"[{TAG}] Visibility-quantile error correlation: {corr['correlation'].item():.6f}")
                results["error_correlation"].update(corr)

        if pmt_ids and has_v:
            results["spatial_slices"] = {}
            for pid in pmt_ids:
                print(f"[{TAG}] Building spatial slice diagnostics for PMT {pid}...")
                results["spatial_slices"][pid] = build_spatial_slice_diagnostics(
                    all_positions_tensor.numpy(),
                    all_pred_vis_tensor.numpy(),
                    all_target_vis_tensor.numpy(),
                    pmt_id=pid,
                )
        elif pmt_ids and rank == 0:
            print(f"[{TAG}] Skipping spatial-slice diagnostics: this model has no "
                  "v prediction to compare against.")

        if do_spectra:
            positions_np = all_positions_tensor.numpy()
            target_vis_np = all_target_vis_tensor.numpy() if has_v else None
            pred_quantiles_np = gathered["pred_quantiles_slice"].numpy() if has_q else None
            target_quantiles_np = gathered["target_quantiles_slice"].numpy() if has_q else None
            quantiles_valid_np = gathered["quantiles_valid_slice"].numpy() if "quantiles_valid_slice" in gathered else None
            grad_targets = load_grad_targets(grad_cache_file, positions_np.shape[0], "quantiles", TAG)

            def fields_for_pmt(i, pid):
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
                return fields, valid

            add_power_spectra(results, positions_np, pmt_ids, fields_for_pmt,
                              spectra_components, spectra_x_margin, TAG)

        save_results(results, output_file, TAG)

    if is_distributed:
        dist.barrier()

    qlib.close()


if __name__ == "__main__":
    from sirentv.eval.eval import main  # one CLI for every pipeline

    main()
