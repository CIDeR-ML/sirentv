from __future__ import annotations

import torch

from sirentv.utils.registry import Registry
from sirentv.utils.transform import cdf_to_pdf
from sirentv.training.utils import unwrap_net

INFER_FNS = Registry("infer_fns")


def build_infer_fn(cfg, net, dl, device):
    """Build logger.infer_fn: __call__(net, x, target, meta) -> plot dict,
    log_step(pred, target, net, meta) -> (target_log, pred_log), cleanup(net, dl, logger, rank)."""
    infer_cfg = cfg.get("logger", {}).get("infer_fn", {})
    if not infer_cfg or "type" not in infer_cfg:
        return _NoOpInfer()
    return INFER_FNS.build(infer_cfg, default_args=dict(cfg=cfg, net=net, dl=dl, device=device))


class _NoOpInfer:
    def __call__(self, net, x, target, meta=None):
        return None

    def log_step(self, pred, target, net, meta=None):
        return target, pred

    def cleanup(self, net, dl, logger, rank):
        pass


class _PlotInfer:
    """Plots pred vs target for dataset[0] (fixed voxel), PMT 40; subclasses rebuild the CDFs."""

    name = "Infer"

    def _set_plot_sample(self, dl, device):
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
            return self._plot(net, self._plot_x, self._plot_target)
        except Exception as e:
            print(f"[{self.name}] Warning: plot inference failed (likely early training): {e}")
            return None

    def log_step(self, pred, target, net, meta=None):
        net_module = unwrap_net(net)
        pred_log = dict(pred)
        target_log = dict(target)
        pred_log["v_linear"] = net_module._inv_xform_vis(pred["v"])
        target_log["v_linear"] = net_module._inv_xform_vis(target["v"])
        return target_log, pred_log

    def _plot(self, net, x, target, batch_id=0, pmt_id=40):
        net_module = unwrap_net(net)
        with torch.no_grad():
            pred = net_module(x)
        rep = pred[self.rep_key]
        pred_v = pred["v"]

        pred_v_linear = net_module._inv_xform_vis(pred_v[batch_id, :])
        target_v_linear = net_module._inv_xform_vis(target["v"][batch_id, :].to(pred_v.device))

        pred_rep = rep[batch_id:batch_id + 1, :, :]
        target_rep = target[self.rep_key][batch_id:batch_id + 1, :, :].to(rep.device)

        pred_t0_ns = torch.exp(pred["t0"][batch_id:batch_id + 1, :])
        if self._plot_meta is not None and "t0_raw" in self._plot_meta:
            target_t0_ns = self._plot_meta["t0_raw"][batch_id:batch_id + 1, :].to(rep.device)
        else:
            target_t0_ns = torch.exp(target["t0"][batch_id:batch_id + 1, :]).to(rep.device)

        pred_cdf, target_cdf, tick_ns = self._cdfs(pred_rep, target_rep, pred_t0_ns, target_t0_ns)
        pred_cdf_pmt = pred_cdf[0, pmt_id, :]
        target_cdf_pmt = target_cdf[0, pmt_id, :]
        pred_pdf = cdf_to_pdf(pred_cdf_pmt, tick_ns)
        target_pdf = cdf_to_pdf(target_cdf_pmt, tick_ns)
        t_window = torch.arange(0, pred_pdf.shape[-1]) * tick_ns

        pred_t0_val = pred_t0_ns[0, pmt_id]
        target_t0_val = target_t0_ns[0, pmt_id].to(pred_t0_val.device)

        return {
            "x_value": t_window,
            "visibility": torch.stack([target_v_linear, pred_v_linear], dim=-1),
            "pdf": torch.stack([target_pdf, pred_pdf], dim=-1),
            "cdf": torch.stack([target_cdf_pmt, pred_cdf_pmt], dim=-1),
            "t0": torch.stack([target_t0_val, pred_t0_val], dim=-1),
            "position": x[batch_id] if x.dim() > 1 else x,
        }


@INFER_FNS.register_module()
class PCAInfer(_PlotInfer):
    name = "PCAInfer"
    rep_key = "coeffs"

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
        self._set_plot_sample(dl, device)

    def cleanup(self, net, dl, logger, rank):
        self._cplib.close()

    def _cdfs(self, pred_c, target_c, pred_t0_ns, target_t0_ns):
        if self._normalize_coeffs and self._cplib.coeff_std is not None:
            pred_c = self._cplib.denormalize_coeffs(pred_c)
            target_c = self._cplib.denormalize_coeffs(target_c)
        pred_cdf = self._cplib.reconstruct_cdf(pred_c, pred_t0_ns)
        target_cdf = self._cplib.reconstruct_cdf(target_c, target_t0_ns)
        return pred_cdf, target_cdf, self._cplib._tick_ns


@INFER_FNS.register_module()
class QuantileInfer(_PlotInfer):
    name = "QuantileInfer"
    rep_key = "quantiles"

    def __init__(self, cfg, net=None, dl=None, device=None, **kwargs):
        import h5py
        from sirentv.data.quantile import QuantilePLib

        plib_cfg = cfg.get("quantile_plib", cfg.get("photonlib", {}))
        filepath = plib_cfg["filepath"]
        mode = plib_cfg.get("mode", "quantile")
        combine_every_quantile = plib_cfg.get("combine_every_quantile", 1)

        self._qlib = QuantilePLib.load(
            filepath, lazy=True, mode=mode, combine_every_quantile=combine_every_quantile,
            log_quantile_C=plib_cfg.get("log_quantile_C"),
        )
        self._align_t0 = bool(plib_cfg.get("align_quantiles_to_t0", False))
        with h5py.File(filepath, "r") as f:
            self._u_grid = f["u_grid"][::combine_every_quantile].astype("float32")
        self._set_plot_sample(dl, device)

    def cleanup(self, net, dl, logger, rank):
        self._qlib.close()

    def _cdfs(self, pred_q, target_q, pred_t0_ns, target_t0_ns):
        pred_cdf = self._qlib.reconstruct_aligned_cdf(
            self._u_grid, pred_q, t0=pred_t0_ns if self._align_t0 else None)
        target_cdf = self._qlib.reconstruct_aligned_cdf(
            self._u_grid, target_q, t0=target_t0_ns if self._align_t0 else None)
        return pred_cdf, target_cdf, self._qlib._t_max_ns / self._qlib._n_bins
