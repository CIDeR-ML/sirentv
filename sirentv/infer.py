from __future__ import annotations

import torch

from sirentv.utils.registry import Registry
from sirentv.utils.transform import pdf_to_cdf, cdf_to_pdf
from sirentv.training.utils import unwrap_net

INFER_FNS = Registry("infer_fns")


def build_infer_fn(cfg, net, dl, device):
    """Build an inference/plotting object from config.

    Args:
        cfg: Full training config. Reads ``cfg["logger"]["infer_fn"]``.
        net: The compiled/DDP-wrapped model.
        dl: The DataLoader.
        device: torch device.

    Returns:
        An object with __call__(net, x, target, meta) -> dict for plotting,
        log_step(pred, target, net, meta) -> (target_log, pred_log),
        and cleanup(net, dl, logger, rank).
    """
    infer_cfg = cfg.get("logger", {}).get("infer_fn", {})
    if not infer_cfg or "type" not in infer_cfg:
        return _NoOpInfer()
    return INFER_FNS.build(infer_cfg, default_args=dict(cfg=cfg, net=net, dl=dl, device=device))


class _NoOpInfer:
    """Fallback when no infer_fn is configured."""
    def __call__(self, net, x, target, meta=None):
        return None
    def log_step(self, pred, target, net, meta=None):
        return target, pred
    def cleanup(self, net, dl, logger, rank):
        pass


# ---------------------------------------------------------------------------
# Waveform inference (existing infer_single_pos_single_pmt logic)
# ---------------------------------------------------------------------------

def infer_single_pos_single_pmt(net, input_x, target, meta, tick_size, batch_id=0, pmt_id=40):
    """Infer visibility and PDF/CDF for a single position and PMT.

    Args:
        net: Unwrapped SirenTV model.
        input_x: (B, 3) positions.
        target: Target dict (loss keys: v, t, ...).
        meta: Meta dict (v_linear, t_linear, v_mask, ...).
        tick_size: Time tick size in ns.
    """
    was_training = net.training
    net.eval()
    use_CDF = net._use_cdf

    if meta is not None and "v_mask" in meta:
        true_mask = meta["v_mask"].squeeze()
        valid_indices = torch.where(true_mask)[0]
        if len(valid_indices) > 0:
            batch_id = valid_indices[0].item()

    with torch.no_grad():
        pred = net(input_x)
    pred_v_linear = net._inv_xform_vis(pred["v"][batch_id, :])

    t_linear = meta["t_linear"] if meta is not None else target.get("t")
    v_linear = meta["v_linear"] if meta is not None else target.get("v")
    target_v_linear = v_linear[batch_id, :].to(pred_v_linear.device)

    t0s = None
    if "t0" in pred:
        pred_t0 = pred["t0"][batch_id, pmt_id] * tick_size
        if "t0" in target:
            target_t0 = target["t0"][batch_id, pmt_id].to(pred_t0.device)
            t0s = torch.stack([target_t0, pred_t0], dim=-1)

    if use_CDF:
        pred_t_cdf = pred["t"][batch_id, pmt_id, :]
        target_t_cdf = t_linear[batch_id, pmt_id, :].to(pred_t_cdf.device)
        pred_t_pdf = cdf_to_pdf(pred_t_cdf, tick_size)
        target_t_pdf = cdf_to_pdf(target_t_cdf, tick_size)
    else:
        pred_t_pdf = pred["t"][batch_id, pmt_id, :]
        target_t_pdf = t_linear[batch_id, pmt_id, :].to(pred_t_pdf.device)
        pred_t_cdf = pdf_to_cdf(pred_t_pdf)
        target_t_cdf = pdf_to_cdf(target_t_pdf)

    t_window = torch.arange(0, pred_t_pdf.shape[-1]) * tick_size

    output = {
        "x_value": t_window,
        "visibility": torch.stack([target_v_linear, pred_v_linear], dim=-1),
        "pdf": torch.stack([target_t_pdf, pred_t_pdf], dim=-1),
        "cdf": torch.stack([target_t_cdf, pred_t_cdf], dim=-1),
        "t0": t0s,
        "position": input_x[batch_id] if input_x.dim() > 1 else input_x,
    }

    if was_training:
        net.train()
    return output


@INFER_FNS.register_module()
class WaveformInfer:
    """Waveform-mode inference for WandB plotting."""

    def __init__(self, cfg, net=None, dl=None, device=None, **kwargs):
        self.tick_size = cfg.get("photonlib", {}).get("time_tick_size", 0.1)
        self._dl = dl

    def __call__(self, net, x, target, meta=None):
        net_module = unwrap_net(net)
        return infer_single_pos_single_pmt(net_module, x, target, meta, self.tick_size)

    def log_step(self, pred, target, net, meta=None):
        net_module = unwrap_net(net)
        pred_log = dict(pred)
        target_log = dict(target)
        if meta is not None:
            target_log["v_linear"] = meta.get("v_linear", target.get("v"))
            target_log["t_linear"] = meta.get("t_linear", target.get("t"))
        pred_log["v_linear"] = net_module._inv_xform_vis(pred["v"])
        pred_log["t_linear"] = pred.get("t", pred.get("coeffs"))
        return target_log, pred_log

    def cleanup(self, net, dl, logger, rank):
        from sirentv.analysis import get_pred_target, log_pred_target
        if rank == 0 and self._dl is not None:
            # get_pred_target expects dataloader._plib — bridge from Dataset
            dl_for_analysis = self._dl
            if not hasattr(dl_for_analysis, "_plib") and hasattr(dl_for_analysis, "dataset"):
                dl_for_analysis._plib = dl_for_analysis.dataset._plib
            net_module = unwrap_net(net)
            try:
                pred, target = get_pred_target(dl_for_analysis, net_module)
                for k in pred.keys():
                    log_pred_target(pred[k], target[k], name=f"comparison_{k}")
            except Exception as e:
                print(f"[WaveformInfer] Warning: post-training analysis failed: {e}")


@INFER_FNS.register_module()
class PCAInfer:
    """PCA-mode inference for WandB plotting."""

    def __init__(self, cfg, net=None, dl=None, device=None, **kwargs):
        from sirentv.data.compressed import CompressedPLib

        cplib_cfg = cfg["compressed_plib"]
        self._cplib = CompressedPLib.load(
            cplib_cfg["filepath"],
            lazy=False,
            n_components=cplib_cfg.get("n_components"),
        )
        self._normalize_coeffs = bool(cplib_cfg.get("normalize_coeffs", False))
        if self._normalize_coeffs:
            self._cplib.compute_coeff_stats()

        # Fixed plot voxel
        if dl is not None:
            plot_sample = dl.dataset[0]
            self._plot_x = plot_sample["position"].unsqueeze(0).to(device)
            self._plot_target = {k: v.unsqueeze(0).to(device) for k, v in plot_sample["target"].items()}
            self._plot_meta = {k: v.unsqueeze(0).to(device) for k, v in plot_sample.get("meta", {}).items()}
        else:
            self._plot_x = None
            self._plot_target = None
            self._plot_meta = None

    def __call__(self, net, x, target, meta=None):
        if self._plot_x is None:
            return None
        try:
            return self._infer_pca_plot(net, self._plot_x, self._plot_target)
        except Exception as e:
            print(f"[PCAInfer] Warning: plot inference failed (likely early training): {e}")
            return None

    def log_step(self, pred, target, net, meta=None):
        net_module = unwrap_net(net)
        pred_log = dict(pred)
        target_log = dict(target)
        pred_log["v_linear"] = net_module._inv_xform_vis(pred["v"])
        target_log["v_linear"] = net_module._inv_xform_vis(target["v"])
        return target_log, pred_log

    def cleanup(self, net, dl, logger, rank):
        self._cplib.close()

    def _infer_pca_plot(self, net, x, target, batch_id=0, pmt_id=40):
        """Reconstruct CDF/PDF from PCA predictions for plotting."""
        net_module = unwrap_net(net)
        net_module.freeze_all()

        with torch.no_grad():
            pred = net_module(x)
        pred_coeffs = pred["coeffs"]
        pred_log_t0 = pred["t0"]
        pred_v = pred["v"]

        pred_v_linear = net_module._inv_xform_vis(pred_v[batch_id, :])
        target_v_linear = net_module._inv_xform_vis(target["v"][batch_id, :].to(pred_v.device))

        pred_c = pred_coeffs[batch_id:batch_id + 1, :, :]
        target_c = target["coeffs"][batch_id:batch_id + 1, :, :].to(pred_coeffs.device)
        if self._normalize_coeffs and self._cplib.coeff_std is not None:
            pred_c = self._cplib.denormalize_coeffs(pred_c)
            target_c = self._cplib.denormalize_coeffs(target_c)

        pred_t0_ns = torch.exp(pred_log_t0[batch_id:batch_id + 1, :])
        # Use raw t0 from meta if available, otherwise exponentiate log-space target
        if self._plot_meta is not None and "t0_raw" in self._plot_meta:
            target_t0_ns = self._plot_meta["t0_raw"][batch_id:batch_id + 1, :].to(pred_coeffs.device)
        else:
            target_t0_ns = torch.exp(target["t0"][batch_id:batch_id + 1, :]).to(pred_coeffs.device)

        pred_cdf = self._cplib.reconstruct_cdf(pred_c, pred_t0_ns)
        target_cdf = self._cplib.reconstruct_cdf(target_c, target_t0_ns)

        tick_ns = self._cplib._tick_ns
        pred_cdf_pmt = pred_cdf[0, pmt_id, :]
        target_cdf_pmt = target_cdf[0, pmt_id, :]

        pred_pdf = cdf_to_pdf(pred_cdf_pmt, tick_ns)
        target_pdf = cdf_to_pdf(target_cdf_pmt, tick_ns)

        t_window = torch.arange(0, pred_pdf.shape[-1]) * tick_ns

        pred_t0_val = pred_t0_ns[0, pmt_id]
        target_t0_val = target_t0_ns[0, pmt_id].to(pred_t0_val.device)
        t0s = torch.stack([target_t0_val, pred_t0_val], dim=-1)

        output = {
            "x_value": t_window,
            "visibility": torch.stack([target_v_linear, pred_v_linear], dim=-1),
            "pdf": torch.stack([target_pdf, pred_pdf], dim=-1),
            "cdf": torch.stack([target_cdf_pmt, pred_cdf_pmt], dim=-1),
            "t0": t0s,
            "position": x[batch_id] if x.dim() > 1 else x,
        }

        net_module.unfreeze_all()
        return output


@INFER_FNS.register_module()
class QuantileInfer:
    """Quantile-mode inference for WandB plotting."""

    def __init__(self, cfg, net=None, dl=None, device=None, **kwargs):
        import h5py
        from sirentv.data.quantile import QuantilePLib

        plib_cfg = cfg.get("quantile_plib", cfg.get("photonlib", {}))
        filepath = plib_cfg["filepath"]
        mode = plib_cfg.get("mode", "quantile")
        combine_every_quantile = plib_cfg.get("combine_every_quantile", 1)

        self._qlib = QuantilePLib.load(
            filepath, lazy=True, mode=mode, combine_every_quantile=combine_every_quantile,
        )
        with h5py.File(filepath, "r") as f:
            self._u_grid = f["u_grid"][::combine_every_quantile].astype("float32")

        if dl is not None:
            plot_sample = dl.dataset[0]
            self._plot_x = plot_sample["position"].unsqueeze(0).to(device)
            self._plot_target = {k: v.unsqueeze(0).to(device) for k, v in plot_sample["target"].items()}
            self._plot_meta = {k: v.unsqueeze(0).to(device) for k, v in plot_sample.get("meta", {}).items()}
        else:
            self._plot_x = None
            self._plot_target = None
            self._plot_meta = None

    def __call__(self, net, x, target, meta=None):
        if self._plot_x is None:
            return None
        try:
            return self._infer_quantile_plot(net, self._plot_x, self._plot_target)
        except Exception as e:
            print(f"[QuantileInfer] Warning: plot inference failed (likely early training): {e}")
            return None

    def log_step(self, pred, target, net, meta=None):
        net_module = unwrap_net(net)
        pred_log = dict(pred)
        target_log = dict(target)
        pred_log["v_linear"] = net_module._inv_xform_vis(pred["v"])
        target_log["v_linear"] = net_module._inv_xform_vis(target["v"])
        return target_log, pred_log

    def cleanup(self, net, dl, logger, rank):
        self._qlib.close()

    def _infer_quantile_plot(self, net, x, target, batch_id=0, pmt_id=40):
        """Reconstruct CDF/PDF from quantile predictions for plotting."""
        net_module = unwrap_net(net)
        net_module.freeze_all()

        with torch.no_grad():
            pred = net_module(x)
        pred_quantiles = pred["quantiles"]
        pred_log_t0 = pred["t0"]
        pred_v = pred["v"]

        pred_v_linear = net_module._inv_xform_vis(pred_v[batch_id, :])
        target_v_linear = net_module._inv_xform_vis(target["v"][batch_id, :].to(pred_v.device))

        pred_q = pred_quantiles[batch_id:batch_id + 1, :, :]
        target_q = target["quantiles"][batch_id:batch_id + 1, :, :].to(pred_quantiles.device)

        pred_cdf = self._qlib.reconstruct_aligned_cdf(self._u_grid, pred_q)
        target_cdf = self._qlib.reconstruct_aligned_cdf(self._u_grid, target_q)

        pred_t0_ns = torch.exp(pred_log_t0[batch_id:batch_id + 1, :])
        if self._plot_meta is not None and "t0_raw" in self._plot_meta:
            target_t0_ns = self._plot_meta["t0_raw"][batch_id:batch_id + 1, :].to(pred_quantiles.device)
        else:
            target_t0_ns = torch.exp(target["t0"][batch_id:batch_id + 1, :]).to(pred_quantiles.device)

        tick_ns = self._qlib._t_max_ns / self._qlib._n_bins
        pred_cdf_pmt = pred_cdf[0, pmt_id, :]
        target_cdf_pmt = target_cdf[0, pmt_id, :]

        pred_pdf = cdf_to_pdf(pred_cdf_pmt, tick_ns)
        target_pdf = cdf_to_pdf(target_cdf_pmt, tick_ns)

        t_window = torch.arange(0, pred_pdf.shape[-1]) * tick_ns

        pred_t0_val = pred_t0_ns[0, pmt_id]
        target_t0_val = target_t0_ns[0, pmt_id].to(pred_t0_val.device)
        t0s = torch.stack([target_t0_val, pred_t0_val], dim=-1)

        output = {
            "x_value": t_window,
            "visibility": torch.stack([target_v_linear, pred_v_linear], dim=-1),
            "pdf": torch.stack([target_pdf, pred_pdf], dim=-1),
            "cdf": torch.stack([target_cdf_pmt, pred_cdf_pmt], dim=-1),
            "t0": t0s,
            "position": x[batch_id] if x.dim() > 1 else x,
        }

        net_module.unfreeze_all()
        return output
