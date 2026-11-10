"""Reader and dataset for the PCA-compressed photon library H5 (vis, t0, PCA coeffs)."""

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from slar.transform import partial_xform_vis
from photonlib.meta import VoxelMeta

from sirentv.data.builder import DATASETS
from sirentv.data.common import grad_target_options, precompute_grad_targets, select_voxel_indices


class CompressedPLib:
    """Reader for the compressed waveform photon library H5 (vis, t0, PCA coeffs)."""

    def __init__(self, filepath, lazy=True, n_components=None, device=None):
        self._path = filepath
        self._lazy = lazy
        self._device = device or torch.device("cpu")
        self._file = None

        with h5py.File(filepath, "r") as f:
            self._n_voxels = f["vis"].shape[0]
            self._n_pmts = f["vis"].shape[1]
            self._n_pca_stored = f["coeffs"].shape[2]
            self._align_window = int(f.attrs.get("align_window", 700))
            self._align_margin = int(f.attrs.get("align_margin", 5))
            self._onset_frac = float(f.attrs.get("onset_frac", 1e-6))
            self._vis_threshold = float(f.attrs.get("vis_threshold", 100))
            self._log_cdf_eps = float(f.attrs.get("log_cdf_eps", 1e-10))
            raw_mode = f.attrs.get("mode", None)
            if raw_mode is not None:
                self._mode = str(raw_mode)
            else:
                self._mode = "log_cdf" if bool(f.attrs.get("log_cdf", False)) else "cdf"
            self._log_cdf = self._mode in ("log_cdf", "log_pdf")

            self._u_grid = None
            self._log_quantile_C = float(f.attrs.get("log_quantile_C", 1e-2))
            self._t_max_ns = float(f.attrs.get("t_max_ns", 600.0))
            self._t0_in_ns = (
                f.attrs.get("t0_in_ns", False) or self._mode in ("log_quantile", "quantile")
                or f["t0"].dtype.kind == "f"
            )
            self._tick_ns = float(f.attrs.get("t_max_ns", 600.0)) / 1000.0
            if self._mode in ("log_quantile", "quantile") and "u_grid" in f:
                self._u_grid = np.asarray(f["u_grid"][:])

            self.pca_mean = torch.from_numpy(f["pca_mean"][:]).float()
            self.pca_components = torch.from_numpy(f["pca_components"][:]).float()
            self.pos = np.asarray(f["pos"][:])
            self.pmt_pos = np.asarray(f["pmt_pos"][:])
            numvox = torch.as_tensor(f["numvox"][:], dtype=torch.float32)
            min_xyz = torch.as_tensor(f["min"][:], dtype=torch.float32)
            max_xyz = torch.as_tensor(f["max"][:], dtype=torch.float32)
            ranges = torch.column_stack((min_xyz, max_xyz))
            self.meta = VoxelMeta(numvox, ranges)

            if "pca_explained_variance_ratio" in f:
                self.explained_variance_ratio = np.asarray(f["pca_explained_variance_ratio"][:])
            else:
                self.explained_variance_ratio = None

        self._n_components = n_components if n_components is not None else self._n_pca_stored
        self._n_components = min(self._n_components, self._n_pca_stored)
        self.pca_mean = self.pca_mean.to(self._device)
        self.pca_components = self.pca_components[: self._n_components].to(self._device)

        self.coeff_std = None
        self.coeff_mean = None

        self.clear_prefetch()

        if lazy:
            self._file = h5py.File(filepath, "r", swmr=True, libver="latest")
            self._vis = self._file["vis"]
            self._t0 = self._file["t0"]
            self._coeffs = self._file["coeffs"]
            self.vis = None
            self.t0 = None
            self.coeffs = None
        else:
            self._file = None
            with h5py.File(filepath, "r") as f:
                self.vis = torch.from_numpy(f["vis"][:]).float()
                t0_arr = f["t0"][:]
                if self._t0_in_ns:
                    self.t0 = torch.from_numpy(t0_arr).float()
                else:
                    self.t0 = torch.from_numpy(t0_arr).long().clamp(min=0)
                self.coeffs = torch.from_numpy(f["coeffs"][:, :, : self._n_components]).float()
            self._vis = self._t0 = self._coeffs = None

    def compute_coeff_stats(self, chunk_size=5000, weight_power: int = 1):
        """Per-component mean/std over all (voxel, PMT) pairs, and coeff_weight_k = K std_k^p / sum_j std_j^p.

        With normalize_coeffs the net penalty on the raw coefficient error is std_k^(p-2):
        p=2 is the unnormalized loss (= reconstruction error, orthonormal components),
        p=0 plain normalized MSE, p=1 (default) a 1/std penalty.
        """
        K = self._n_components
        with h5py.File(self._path, "r") as f:
            ds = f["coeffs"]
            n_vox = ds.shape[0]
            n_tot = n_vox * ds.shape[1]
            sum_c = np.zeros(K, dtype=np.float64)
            sum_sq = np.zeros(K, dtype=np.float64)
            for s in range(0, n_vox, chunk_size):
                c = np.asarray(ds[s : min(s + chunk_size, n_vox), :, :K], dtype=np.float64)
                c = c.reshape(-1, K)
                sum_c += c.sum(axis=0)
                sum_sq += (c ** 2).sum(axis=0)
        mean = sum_c / n_tot
        std = np.sqrt(np.maximum(sum_sq / n_tot - mean ** 2, 0.0))
        std = np.maximum(std, 1e-12)
        self.coeff_mean = torch.from_numpy(mean.astype(np.float32)).to(self._device)
        self.coeff_std = torch.from_numpy(std.astype(np.float32)).to(self._device)
        # mean weight 1, so the loss scale matches an unweighted run
        std_pow = self.coeff_std ** weight_power
        self.coeff_weight = self.coeff_std.numel() * std_pow / std_pow.sum()
        return self.coeff_mean, self.coeff_std

    def normalize_coeffs(self, coeffs):
        """Normalize coefficients to zero-mean, unit-variance per component."""
        if self.coeff_std is None:
            raise RuntimeError("call compute_coeff_stats() first")
        m = self.coeff_mean.to(coeffs.device)
        s = self.coeff_std.to(coeffs.device)
        return (coeffs - m) / s

    def denormalize_coeffs(self, coeffs_norm):
        """Inverse of normalize_coeffs."""
        if self.coeff_std is None:
            raise RuntimeError("call compute_coeff_stats() first")
        m = self.coeff_mean.to(coeffs_norm.device)
        s = self.coeff_std.to(coeffs_norm.device)
        return coeffs_norm * s + m

    def __len__(self):
        return self._n_voxels

    @classmethod
    def load(cls, filepath, lazy=True, n_components=None, device=None):
        return cls(filepath, lazy=lazy, n_components=n_components, device=device)

    def _read_slice(self, key, voxel_ids):
        if self._file is None:
            if isinstance(voxel_ids, (int, np.integer)):
                return getattr(self, key)[voxel_ids]
            return getattr(self, key)[voxel_ids].clone()
        ds = getattr(self, f"_{key}")
        if isinstance(voxel_ids, (int, np.integer)):
            arr = ds[voxel_ids]
        else:
            voxel_ids = np.asarray(voxel_ids)
            if voxel_ids.ndim == 0:
                arr = ds[int(voxel_ids)]
            else:
                uniq, inv = np.unique(voxel_ids, return_inverse=True)
                arr = ds[uniq][inv]
        return torch.from_numpy(np.asarray(arr)).to(self._device)

    def __getitem__(self, voxel_ids):
        if self._file is None and not isinstance(voxel_ids, (int, np.integer)):
            vox = voxel_ids
            if isinstance(vox, np.ndarray):
                vox = torch.from_numpy(vox).long().to(self.vis.device)
            uniq, inv = torch.unique(vox, return_inverse=True)
            # index_select can be faster than [uniq] for large plibs (gather from huge tensor)
            vis = torch.index_select(self.vis, 0, uniq)[inv].clone()
            t0 = torch.index_select(self.t0, 0, uniq)[inv].clone()
            coeffs = torch.index_select(self.coeffs, 0, uniq)[inv].clone()
        else:
            vis = self._read_slice("vis", voxel_ids)
            t0 = self._read_slice("t0", voxel_ids)
            coeffs = self._read_slice("coeffs", voxel_ids)
        if coeffs.dim() == 2:
            coeffs = coeffs[:, : self._n_components]
        else:
            coeffs = coeffs[..., : self._n_components]
        return {"vis": vis, "t0": t0, "coeffs": coeffs}

    def clear_prefetch(self):
        self._event_uniq = None
        self._event_data = None

    def prefetch_event(self, positions, device=None):
        """Cache every field for the unique voxels of `positions` (N, 3), so lookup() on those
        points reads memory instead of HDF5. None clears the cache."""
        if positions is None:
            self.clear_prefetch()
            return
        dev = device if device is not None else self._device
        vox = self.meta.coord_to_voxel(np.atleast_2d(np.asarray(positions, dtype=np.float64))).numpy()
        self._event_uniq = np.unique(vox)
        data = self[self._event_uniq]
        self._event_data = {k: v.to(dev) for k, v in data.items()}

    def lookup(self, pos):
        """pos: (N, 3) -> {vis, t0, coeffs}, from the prefetch cache when one is set."""
        vox = self.meta.coord_to_voxel(np.atleast_2d(np.asarray(pos, dtype=np.float64))).numpy()
        if self._event_uniq is not None:
            inv = torch.from_numpy(np.searchsorted(self._event_uniq, vox)).long()
            return {k: v[inv.to(v.device)].clone() for k, v in self._event_data.items()}
        if self._file is None:
            vox = torch.from_numpy(vox).long().to(self.vis.device)
        return self[vox]

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def reconstruct_aligned_cdf(self, coeffs):
        """coeffs: (..., K) -> aligned CDF (..., align_window).

        Handles all modes:
          cdf      — raw output is CDF
          log_cdf  — exp to get CDF
          pdf      — clamp+cumsum to get CDF
          log_pdf  — exp then cumsum to get CDF
          anscombe — inverse Anscombe to get PDF, then cumsum
          log_quantile / quantile — 511-dim quantile -> 1000-bin CDF via interp
        """
        if self._mode in ("log_quantile", "quantile"):
            return self._reconstruct_aligned_cdf_quantile(coeffs)
        recon = self.reconstruct_aligned_raw(coeffs)
        if self._mode == "log_cdf":
            return torch.exp(recon)
        elif self._mode == "pdf":
            return recon.clamp(min=0).cumsum(dim=-1)
        elif self._mode == "log_pdf":
            return torch.exp(recon).cumsum(dim=-1)
        elif self._mode == "anscombe":
            pdf = (recon / 2.0) ** 2 - 3.0 / 8.0
            return pdf.clamp(min=0).cumsum(dim=-1)
        return recon

    def _reconstruct_aligned_cdf_quantile(self, coeffs):
        """coeffs (..., K) -> aligned CDF (..., 1000) from quantile PCA."""
        if self._u_grid is None:
            raise RuntimeError("quantile mode but no u_grid in file")
        raw = self.reconstruct_aligned_raw(coeffs)
        device = raw.device
        dtype = raw.dtype
        u_np = self._u_grid.astype(np.float32)
        C = self._log_quantile_C
        if self._mode == "log_quantile":
            q_a = np.power(10.0, raw.detach().cpu().numpy()).clip(min=1e-30) - C
            q_a = np.clip(q_a, 0.0, None)
        else:
            q_a = np.clip(raw.detach().cpu().numpy(), 0.0, None)
        q_full = q_a  # full 512 quantile times (aligned to t0)
        t_edges = np.linspace(0, self._t_max_ns, 1001, dtype=np.float32)
        batch_shape = raw.shape[:-1]
        flat = q_full.reshape(-1, q_full.shape[-1])
        f_flat = np.array(
            [np.interp(t_edges, flat[i], u_np, left=0.0, right=1.0)[:-1] for i in range(flat.shape[0])],
            dtype=np.float32,
        )
        out = torch.from_numpy(f_flat.reshape(*batch_shape, 1000)).to(device=device, dtype=dtype)
        return out

    def to_linear_time(self, coeffs):
        """coeffs: (..., K) -> raw linear-ns quantile times (..., n_quantile), the PCA-
        reconstructed quantile function evaluated at self._u_grid, without interpolating
        onto a fixed CDF grid. Only valid when self._mode is 'quantile' or 'log_quantile'.
        """
        if self._mode not in ("log_quantile", "quantile"):
            raise RuntimeError(f"to_linear_time only valid for quantile/log_quantile mode, got {self._mode!r}")
        raw = self.reconstruct_aligned_raw(coeffs)
        if self._mode == "log_quantile":
            q = torch.pow(10.0, raw) - self._log_quantile_C
        else:
            q = raw
        return q.clamp(min=0.0)

    def reconstruct_aligned_raw(self, coeffs):
        """coeffs: (..., K) -> raw PCA output (..., align_window), no post-processing.

        Returns the reconstruction in whatever space PCA was trained in
        (CDF, log-CDF, PDF, or log-PDF). Use reconstruct_aligned_cdf for the
        actual CDF regardless of mode.
        """
        coeffs = coeffs.to(self._device)
        if coeffs.shape[-1] > self._n_components:
            coeffs = coeffs[..., : self._n_components]
        mean = self.pca_mean.to(coeffs.device)
        comp = self.pca_components.to(coeffs.device)
        return mean + coeffs @ comp

    @property
    def mode(self):
        """Compression mode: 'cdf', 'log_cdf', 'pdf', or 'log_pdf'."""
        return self._mode

    @property
    def log_cdf(self):
        """Whether the PCA basis uses a log transform (log_cdf or log_pdf)."""
        return self._log_cdf

    def reconstruct_cdf(self, coeffs, t0, n_bins=1000):
        """
        Reconstruct full CDF in original time bins.
        coeffs: (..., K), t0: (...) int, same leading shape.
        Returns (..., n_bins).
        """
        W = self._align_window
        M = self._align_margin
        aligned = self.reconstruct_aligned_cdf(coeffs)
        need_squeeze = False
        if aligned.dim() == 2:
            aligned = aligned.unsqueeze(0)
            t0 = torch.atleast_1d(t0).unsqueeze(0)
            need_squeeze = True
        elif t0.dim() == 1 and aligned.dim() == 3:
            t0 = t0.unsqueeze(-1).expand_as(aligned[..., :1]).squeeze(-1)

        if t0.is_floating_point():
            t0 = (t0 / self._tick_ns).round().long().clamp(min=0)
        else:
            t0 = t0.long().clamp(min=0)
        batch_shape = aligned.shape[:-1]
        out = torch.zeros(*batch_shape, n_bins, device=aligned.device, dtype=aligned.dtype)
        # aligned[k] corresponds to CDF bin (k + t0 - M)
        # so CDF[bin] = aligned[bin - t0 + M]
        for i in range(aligned.shape[0]):
            for j in range(aligned.shape[1]):
                t0_ij = t0[i, j].item()
                a_lo = max(0, t0_ij - M)
                a_hi = min(n_bins, t0_ij - M + W)
                buf_lo = a_lo + M - t0_ij
                buf_hi = buf_lo + (a_hi - a_lo)
                out[i, j, a_lo:a_hi] = aligned[i, j, int(buf_lo) : int(buf_hi)]
                out[i, j, a_hi:] = 1.0
        if need_squeeze:
            out = out.squeeze(0)
        return out


@DATASETS.register_module()
class CompressedPLibDataset(Dataset):
    """Map-style dataset over CompressedPLib: position -> target (v, t0, coeffs[, coeffs_weight, *_grad_frob]).

    v = xform_vis(vis / n_photon), t0 = log(t0), coeffs normalized when normalize_coeffs.
    meta carries v_linear (raw counts), t0_raw and voxel_id. photonlib.device: cuda or
    lazy: false preloads every target; otherwise each batch is one HDF5 read per field.
    """

    def __init__(self, cfg, rank=0, world_size=1):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        plib_cfg = cfg.get("compressed_plib", cfg.get("photonlib", {}))
        filepath = plib_cfg.get("filepath")
        lazy = plib_cfg.get("lazy", True)
        n_components = plib_cfg.get("n_components")
        self._vis_eps = plib_cfg.get("vis_eps", 1e-6)
        self._n_photon = float(plib_cfg.get("n_photon", 1.0))
        self._device = torch.device(plib_cfg.get("device", "cpu"))
        self._on_gpu = self._device.type == "cuda"

        self._plib = CompressedPLib.load(filepath, lazy=(lazy and not self._on_gpu), n_components=n_components)

        self._normalize_coeffs = bool(plib_cfg.get("normalize_coeffs", False))
        if self._normalize_coeffs:
            coeff_weight_power = plib_cfg.get("weight_power", 1)
            print("[CompressedPLibDataset] computing per-component coeff stats for normalization "
                  f"(weight_power={coeff_weight_power}) ...")
            self._plib.compute_coeff_stats(weight_power=coeff_weight_power)
            print(f"  coeff_std range: [{self._plib.coeff_std.min():.4g}, {self._plib.coeff_std.max():.4g}]")
            print(f"  coeff_weight range: [{self._plib.coeff_weight.min():.4g}, {self._plib.coeff_weight.max():.4g}]")

        self._xform_vis, self._inv_xform_vis = partial_xform_vis(cfg.get("transform_vis", {}))

        self.indices = select_voxel_indices(
            cfg.get("data", {}), len(self._plib), self._plib.pos, rank, "CompressedPLibDataset")

        train_cfg = cfg.get("train", {})
        self._grad_keys = [k for k in train_cfg.get("grad_supervision_keys", []) if k in ("v", "coeffs")]
        self._grad_opts = grad_target_options(train_cfg)
        if self._grad_keys:
            self._grad_targets = precompute_grad_targets(
                self._plib, self.indices.numpy(), self._grad_keys, "coeffs",
                self._read_v, self._read_coeffs, self._plib._n_components,
                train_cfg, self._grad_opts, self._grad_cache_meta(), self._n_photon,
                "CompressedPLibDataset",
            )

        if self._on_gpu or not lazy:
            self._precompute_targets()

    def _grad_cache_meta(self):
        """Every config choice the cached gradient targets depend on (a mismatch recomputes them)."""
        from sirentv.utils.misc import GRAD_CACHE_FORMAT_VERSION

        xform_cfg = self.cfg.get("transform_vis", {})
        plib_cfg = self.cfg.get("compressed_plib", self.cfg.get("photonlib", {}))
        return {
            "format_version": GRAD_CACHE_FORMAT_VERSION,
            "lut_filepath": plib_cfg.get("filepath"),
            "total_voxels": len(self._plib),
            "n_photon": self._n_photon,
            "xform_vmax": xform_cfg.get("vmax", 1.0),
            "xform_eps": xform_cfg.get("eps", 1e-8),
            "xform_sin_out": xform_cfg.get("sin_out", False),
            "normalize_coeffs": self._normalize_coeffs,
            "coeff_weight_power": plib_cfg.get("weight_power", 1) if self._normalize_coeffs else None,
            "n_components": self._plib._n_components,
            "n_projections": self._grad_opts["n_projections"],
            "seed": 0,
        }

    def _read_v(self, rows):
        vis_norm = self._plib[rows]["vis"].float() / self._n_photon
        return self._xform_vis(vis_norm).numpy()[:, :, None]  # (chunk, n_pmt, 1)

    def _read_coeffs(self, rows):
        coeffs = self._plib[rows]["coeffs"].float()
        if self._normalize_coeffs:
            coeffs = self._plib.normalize_coeffs(coeffs)
        # invalid pairs are already masked out of the finite difference; this keeps NaN out of logs
        return torch.nan_to_num(coeffs, nan=0.0).numpy()  # (chunk, n_pmt, K)

    def _targets(self, data):
        """Raw LUT fields -> (v, t0, coeffs) in the network's target domain."""
        vis_raw = data["vis"].float()
        t0_raw = data["t0"].float()
        coeffs = data["coeffs"].float()
        if self._normalize_coeffs:
            coeffs = self._plib.normalize_coeffs(coeffs)
        v = self._xform_vis(vis_raw / self._n_photon)
        t0 = torch.log(t0_raw.clamp(min=1e-3))
        return vis_raw, t0_raw, v, t0, coeffs

    def _precompute_targets(self):
        """Store every transformed target on self._device."""
        idx = self.indices.numpy()
        self._pos = torch.from_numpy(self._plib.pos[idx]).float().to(self._device)
        vis_raw, t0_raw, v, t0, coeffs = self._targets(self._plib[idx])
        self._v = v.to(self._device)
        self._t0 = t0.to(self._device)
        self._coeffs = coeffs.to(self._device)
        self._vis_raw = vis_raw.to(self._device)
        self._t0_raw = t0_raw.to(self._device)
        self._plib.close()
        print(f"[CompressedPLibDataset] precomputed {len(self.indices)} voxels on {self._device}")

    def __len__(self):
        return len(self.indices)

    def _coeffs_weight(self):
        # (1, K) per-component loss weight, broadcasts against the collated (B, n_pmt, K)
        return self._plib.coeff_weight.unsqueeze(0).cpu() if self._normalize_coeffs else None

    def _sample(self, pos, v, t0, coeffs, vis_raw, t0_raw, voxel_id, grads, coeffs_weight):
        target = {"v": v, "t0": t0, "coeffs": coeffs}
        if coeffs_weight is not None:
            target["coeffs_weight"] = coeffs_weight
        for key in self._grad_keys:
            target[f"{key}_grad_frob"] = grads[key]
        return {
            "position": pos,
            "target": target,
            "meta": {"v_linear": vis_raw, "t0_raw": t0_raw, "voxel_id": voxel_id},
        }

    def __getitem__(self, idx):
        if hasattr(self, "_v"):
            return self._sample(
                self._pos[idx], self._v[idx], self._t0[idx], self._coeffs[idx],
                self._vis_raw[idx], self._t0_raw[idx], self.indices[idx],
                {k: self._grad_targets[k][idx] for k in self._grad_keys}, self._coeffs_weight(),
            )
        vox_id = self.indices[idx].item()
        pos = torch.from_numpy(self._plib.pos[vox_id]).float()
        vis_raw, t0_raw, v, t0, coeffs = self._targets(self._plib[vox_id])
        return self._sample(pos, v, t0, coeffs, vis_raw, t0_raw, self.indices[idx],
                            {k: self._grad_targets[k][idx] for k in self._grad_keys}, self._coeffs_weight())

    def __getitems__(self, indices):
        """A whole batch with ONE HDF5 read per field (torch >= 2.0 calls this instead of
        per-index __getitem__). Per-voxel reads of the contiguous coeffs are latency-bound
        (~11 MB/s measured); one multi-row selection reaches ~1.2 GB/s. Sampling is unchanged."""
        if hasattr(self, "_v"):
            return [self[i] for i in indices]

        idx = np.asarray(indices)
        # CompressedPLib sorts/deduplicates the ids for h5py
        vox = self.indices[torch.as_tensor(idx, dtype=torch.long)].numpy()
        vis_raw, t0_raw, v, t0, coeffs = self._targets(self._plib[vox])  # (B, n_pmt[, K])
        pos = torch.from_numpy(self._plib.pos[vox]).float()
        coeffs_weight = self._coeffs_weight()
        grads = {k: self._grad_targets[k][idx] for k in self._grad_keys}
        return [
            self._sample(pos[j], v[j], t0[j], coeffs[j], vis_raw[j], t0_raw[j], self.indices[idx[j]],
                         {k: g[j] for k, g in grads.items()}, coeffs_weight)
            for j in range(idx.shape[0])
        ]
