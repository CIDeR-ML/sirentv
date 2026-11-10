from __future__ import annotations

import torch
import torch.distributed as dist
from tqdm import tqdm

from sirentv.data.builder import create_dataloader
from sirentv.data.compressed import CompressedPLib
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
    save_results,
    setup_eval_model,
)

TAG = "eval_pca"
VALID_COMPARISONS = {"pred_vs_compressed", "pred_vs_raw", "compressed_vs_raw"}


@torch.no_grad()
def evaluate_pca(
    cfg: dict,
    output_file: str = "eval_pca_results.pt",
    pmt_ids: list[int] | None = None,
    ckpt_file: str | None = None,
    comparisons: list[str] | None = None,
    spectra: bool = True,
    spectra_components: list[int] | None = None,
    spectra_x_margin: int = 20,
    grad_cache_file: str | None = None,
):
    """Evaluate a PCA model in native quantile bins. Comparisons:
    pred_vs_compressed (model error vs. its training target), compressed_vs_raw (PCA ceiling),
    pred_vs_raw (end to end). The last two need raw_quantile_plib, the slow VDS-backed file;
    the default runs them only when it is configured."""
    rank, world_size, device, net_module = setup_eval_model(cfg, ckpt_file, TAG)
    is_distributed = world_size > 1

    dl = create_dataloader(eval_loader_cfg(cfg), rank=rank, world_size=world_size)

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
        print(f"[{TAG}] Comparisons: {sorted(comparisons)}")

    if compute_pca_ceiling:
        raw_qlib = QuantilePLib.load(
            raw_plib_cfg["filepath"], lazy=True, mode=raw_plib_cfg.get("mode", "quantile"),
            combine_every_quantile=raw_plib_cfg.get("combine_every_quantile", 1),
        )
        if do_compressed_vs_raw:
            acc_compressed_vs_raw = PairwiseBiasAccumulator(n_pmts, n_quantile, threshold, device)
        if do_pred_vs_raw:
            acc_pred_vs_raw = PairwiseBiasAccumulator(n_pmts, n_quantile, threshold, device)
        if rank == 0:
            print(f"[{TAG}] Loading raw quantile file: {raw_plib_cfg['filepath']}")

    if rank == 0:
        print(f"[{TAG}] n_pmts: {n_pmts}, n_quantile: {n_quantile}, normalize_coeffs: {normalize_coeffs}")

    acc_pred_vs_compressed = PairwiseBiasAccumulator(n_pmts, n_quantile, threshold, device)
    vis_stats = VisStatsAccumulator(n_pmts, threshold, device)
    t0_acc_pred_vs_compressed = ScalarErrorAccumulator(device)
    # compressed-target adjacent-quantile spacing, for a density-corrected Poisson floor
    spacing_acc = BinnedMeanAccumulator(n_quantile - 1, device)

    # coefficient spectra accumulate only the requested PMTs/components: the full (N, 81, 50)
    # array is what OOM-killed the interactive version of this diagnostic
    do_spectra = bool(spectra and pmt_ids)
    spectra_components = list(spectra_components) if spectra_components else [0, 1, 2, 3, 4]
    if do_spectra and rank == 0:
        print(f"[{TAG}] Power spectra enabled for PMTs {pmt_ids}, "
              f"coeff components {spectra_components}")

    # per-batch results, kept on CPU for the whole loop
    collected = {k: [] for k in (
        "pred_vis", "compressed_vis", "pred_t0", "compressed_t0", "positions",
        "vis_errors", "quantile_errors", "pred_coeffs_slice", "compressed_coeffs_slice",
        "time_bias", "ceiling_time_bias",
    )}

    for data in tqdm(dl, desc=f"Rank {rank} eval", disable=(rank != 0)):
        x = data["position"].contiguous().to(device)
        target = {k: v.contiguous().to(device) for k, v in data["target"].items()}

        pred_out: dict[str, torch.Tensor] = net_module(x)

        pred_v_linear = net_module._inv_xform_vis(pred_out["v"])  # (B, n_pmts)
        compressed_v_linear = net_module._inv_xform_vis(target["v"])

        # both sides are log(t0)
        pred_t0 = torch.exp(pred_out["t0"])  # (B, n_pmts)
        compressed_t0 = torch.exp(target["t0"])

        pred_coeffs = pred_out["coeffs"]
        compressed_coeffs = target["coeffs"]
        if normalize_coeffs:
            pred_coeffs = cplib.denormalize_coeffs(pred_coeffs)
            compressed_coeffs = cplib.denormalize_coeffs(compressed_coeffs)

        pred_time = cplib.to_linear_time(pred_coeffs)  # (B, n_pmts, n_quantile), t0-relative
        compressed_time = cplib.to_linear_time(compressed_coeffs)

        if do_pred_vs_compressed:
            acc_pred_vs_compressed.update(pred_v_linear, compressed_v_linear, pred_time, compressed_time)

        if compute_pca_ceiling:
            voxel_id = data["meta"]["voxel_id"].numpy()  # (B,)
            raw_quantiles = raw_qlib[voxel_id]["quantiles"].to(device)
            # zero-visibility pairs have NaN quantiles; they are below the visibility threshold anyway
            raw_quantiles = torch.nan_to_num(raw_quantiles, nan=0.0)
            raw_time = raw_qlib.to_linear_time(raw_quantiles)  # absolute time

            # The PCA was fit on onset-subtracted times, raw_time is absolute: add t0 back, the
            # true one for the ceiling, the predicted one end to end.
            compressed_time_abs = compressed_time + compressed_t0.unsqueeze(-1)
            pred_time_abs = pred_time + pred_t0.unsqueeze(-1)

            if do_compressed_vs_raw:
                # only the time representation is PCA-lossy: compressed visibility on both sides
                acc_compressed_vs_raw.update(compressed_v_linear, compressed_v_linear, compressed_time_abs, raw_time)
                collected["ceiling_time_bias"].append(
                    masked_bin_bias(compressed_time_abs, raw_time, raw_time > threshold).detach().cpu())

            if do_pred_vs_raw:
                acc_pred_vs_raw.update(pred_v_linear, compressed_v_linear, pred_time_abs, raw_time)

        collected["vis_errors"].append((pred_v_linear - compressed_v_linear).detach().cpu())
        collected["quantile_errors"].append((pred_time - compressed_time).mean(dim=-1).detach().cpu())

        if do_pred_vs_compressed:
            collected["time_bias"].append(
                masked_bin_bias(pred_time, compressed_time, compressed_time > threshold).detach().cpu())

        compressed_vis_mask = vis_stats.update(compressed_v_linear)
        t0_acc_pred_vs_compressed.update(pred_t0, compressed_t0, compressed_vis_mask)
        compressed_dt = compressed_time[..., 1:] - compressed_time[..., :-1]  # (B, n_pmts, n_quantile-1)
        spacing_acc.update(compressed_dt, compressed_vis_mask)

        collected["pred_vis"].append(pred_v_linear.detach().cpu())
        collected["compressed_vis"].append(compressed_v_linear.detach().cpu())
        collected["pred_t0"].append(pred_t0.detach().cpu())
        collected["compressed_t0"].append(compressed_t0.detach().cpu())
        collected["positions"].append(x.detach().cpu())

        if do_spectra:
            # coefficients are already denormalized above
            sel = torch.as_tensor(pmt_ids, device=pred_coeffs.device)
            comp = torch.as_tensor(spectra_components, device=pred_coeffs.device)
            collected["pred_coeffs_slice"].append(pred_coeffs.detach()[:, sel][..., comp].cpu())
            collected["compressed_coeffs_slice"].append(compressed_coeffs.detach()[:, sel][..., comp].cpu())

    local_results = {k: torch.cat(v, dim=0) for k, v in collected.items() if v}

    if is_distributed:
        if do_pred_vs_compressed:
            acc_pred_vs_compressed.all_reduce()
        vis_stats.all_reduce()
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

        t0_stats_pred_vs_compressed = t0_acc_pred_vs_compressed.finalize()

        print(f"[{TAG}] Total positions evaluated: {all_pred_vis_tensor.shape[0]}")
        if do_pred_vs_compressed:
            stats_pred_vs_compressed = acc_pred_vs_compressed.finalize()
            print(f"[{TAG}] [pred vs compressed] Overall visibility bias: {stats_pred_vs_compressed['overall']['vis_bias'].item():.6e}")
            print(f"[{TAG}] [pred vs compressed] Overall quantile-bin bias: {stats_pred_vs_compressed['overall']['time_bias'].item():.6e}")
        print(f"[{TAG}] t0 bias mean: {t0_stats_pred_vs_compressed['mean'].item():.6e}, std: {t0_stats_pred_vs_compressed['std'].item():.6e}")
        if do_compressed_vs_raw:
            stats_compressed_vs_raw = acc_compressed_vs_raw.finalize()
            print(f"[{TAG}] [compressed vs raw, PCA ceiling] Overall quantile-bin bias: {stats_compressed_vs_raw['overall']['time_bias'].item():.6e}")
        if do_pred_vs_raw:
            stats_pred_vs_raw = acc_pred_vs_raw.finalize()
            print(f"[{TAG}] [pred vs raw, end-to-end] Overall quantile-bin bias: {stats_pred_vs_raw['overall']['time_bias'].item():.6e}")

        corr = error_correlation(gathered["vis_errors"], gathered["quantile_errors"])
        print(f"[{TAG}] Visibility-quantile error correlation: {corr['correlation'].item():.6f}")

        results = {
            "compressed_target": {"visibility_per_pmt": vis_stats.finalize()},
            "visibility_all": {
                "pred": all_pred_vis_tensor,
                "compressed_target": all_compressed_vis_tensor,
                "positions": all_positions_tensor,
            },
            "t0_all": {
                "pred": all_pred_t0_tensor,
                "target": all_compressed_t0_tensor,
            },
            # length n_quantile-1; local density f(Q(u_k)) ~= delta_u / quantile_time_spacing[k]
            "quantile_time_spacing": spacing_acc.finalize(),
            "error_correlation": corr,
            "meta": {
                "n_pmts": n_pmts,
                "pmt_pos": torch.as_tensor(cplib.pmt_pos, dtype=torch.float32).cpu(),
                "n_quantile": n_quantile,
                "n_positions": all_pred_vis_tensor.shape[0],
                "x_min": cfg.get("data", {}).get("x_min"),  # [mm]; None = every LUT voxel
                "normalize_coeffs": normalize_coeffs,
                "threshold": threshold,
                # target visibility is vis_raw / n_photon; checkpoints may differ in n_photon
                "n_photon": cplib_cfg.get("n_photon"),
                "wall_ranges": cplib.meta.ranges.cpu(),  # (3, 2) min/max per axis
            },
        }

        if do_pred_vs_compressed:
            results["pred_vs_compressed"] = {
                **stats_pred_vs_compressed,
                "t0_bias": t0_stats_pred_vs_compressed,
            }
            results["error_correlation"]["time_bias"] = gathered["time_bias"]
        if do_compressed_vs_raw:
            results["compressed_vs_raw"] = stats_compressed_vs_raw
            results["error_correlation"]["ceiling_time_bias"] = gathered["ceiling_time_bias"]
        if do_pred_vs_raw:
            results["pred_vs_raw"] = stats_pred_vs_raw

        if pmt_ids:
            results["spatial_slices"] = {}
            for pid in pmt_ids:
                print(f"[{TAG}] Building spatial slice diagnostics for PMT {pid}...")
                results["spatial_slices"][pid] = build_spatial_slice_diagnostics(
                    all_positions_tensor.numpy(),
                    all_pred_vis_tensor.numpy(),
                    all_compressed_vis_tensor.numpy(),
                    pmt_id=pid,
                )

        if do_spectra:
            positions_np = all_positions_tensor.numpy()
            target_vis_np = all_compressed_vis_tensor.numpy()
            pred_coeffs_np = gathered["pred_coeffs_slice"].numpy()
            target_coeffs_np = gathered["compressed_coeffs_slice"].numpy()
            # target side only: a prediction-side gradient needs grad_keys, which no_grad eval skips
            grad_targets = load_grad_targets(grad_cache_file, positions_np.shape[0], "coeffs", TAG)

            def fields_for_pmt(i, pid):
                fields = {
                    "v": {"pred": all_pred_vis_tensor.numpy()[:, pid], "target": target_vis_np[:, pid]},
                    "t0": {"pred": all_pred_t0_tensor.numpy()[:, pid], "target": all_compressed_t0_tensor.numpy()[:, pid]},
                    "coeffs": {"pred": pred_coeffs_np[:, i, :], "target": target_coeffs_np[:, i, :]},
                }
                if grad_targets is not None:
                    fields["grad_v"] = {"target": grad_targets["grad_v"][:, pid]}
                    fields["grad_coeffs"] = {"target": grad_targets["grad_coeffs"][:, pid]}
                return fields, target_vis_np[:, pid] > 0

            add_power_spectra(results, positions_np, pmt_ids, fields_for_pmt,
                              spectra_components, spectra_x_margin, TAG)

        save_results(results, output_file, TAG)

    if is_distributed:
        dist.barrier()

    cplib.close()
    if compute_pca_ceiling:
        raw_qlib.close()


if __name__ == "__main__":
    from sirentv.eval.eval import main  # one CLI for every pipeline

    main()
