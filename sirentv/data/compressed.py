"""
Compressed photon library reader and dataset for (vis, t0, PCA coeffs) format.
"""

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from slar.transform import partial_xform_vis
from photonlib.meta import VoxelMeta

from sirentv.data.builder import DATASETS


class CompressedPLib:
    """
    Reader for compressed waveform photon library H5 (vis, t0, PCA coeffs).
    """

    def __init__(self, filepath, lazy=True, n_components=None, device=None, reflect_x=False, pmt_flip_perm=None):
        self._path = filepath
        self._lazy = lazy
        self._device = device or torch.device("cpu")
        self._file = None
        self._reflect_x = bool(reflect_x)

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

            if self._reflect_x:
                full_min = torch.tensor([min_xyz[0], min_xyz[1], min_xyz[2]])
                full_max = torch.tensor([-min_xyz[0], max_xyz[1], max_xyz[2]])
                full_ranges = torch.column_stack((full_min, full_max))
                self.full_meta = VoxelMeta(numvox, full_ranges)
                if pmt_flip_perm is not None:
                    self._pmt_flip_perm = np.asarray(pmt_flip_perm, dtype=np.int64)
                else:
                    # Chroma-lar 2-wall: same (y,z) order on both sides. If plib has only one wall
                    # (all pmt_pos same sign of x), then +x PMT i has same (y,z) as plib PMT i -> identity.
                    x = self.pmt_pos[:, 0]
                    all_neg = np.all(x <= 0)
                    all_pos = np.all(x >= 0)
                    if all_neg or all_pos:
                        self._pmt_flip_perm = np.arange(len(self.pmt_pos), dtype=np.int64)
                    else:
                        # Both walls in plib: find j at (-px_i, py_i, pz_i) for each i.
                        reflected = self.pmt_pos.copy()
                        reflected[:, 0] *= -1
                        dist = np.sum((self.pmt_pos[:, None, :] - reflected[None, :, :]) ** 2, axis=2)
                        self._pmt_flip_perm = np.argmin(dist, axis=1)
                        max_d = np.sqrt(np.max(np.min(dist, axis=1)))
                        if max_d > 1.0:
                            import warnings
                            warnings.warn(
                                f"reflect_x: auto pmt_flip_perm has max match distance {max_d:.2f} mm; "
                                "consider passing pmt_flip_perm from geometry."
                            )
                        inv = self._pmt_flip_perm[self._pmt_flip_perm]
                        if not np.array_equal(inv, np.arange(len(self._pmt_flip_perm))):
                            import warnings
                            warnings.warn(
                                "reflect_x: pmt_flip_perm is not an involution (perm[perm[i]] != i); "
                                "PMT layout may not be x-symmetric. Pass pmt_flip_perm from geometry."
                            )
            else:
                self.full_meta = None
                self._pmt_flip_perm = None

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

        self._event_uniq = None
        self._event_vis = None
        self._event_t0 = None
        self._event_coeffs = None

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
        """Compute per-component mean and std over all (voxel, PMT) pairs.

        weight_power (p): sets coeff_weight_k = K * std_k^p / sum(std_j^p), i.e. the net
        penalty on the *raw* coefficient error once combined with normalize_coeffs' division
        by std is std_k^(p-2) (see derivation in compute_coeff_stats' caller / training docs):
        p=2 reproduces the original unnormalized loss exactly (no net penalty, since PCA
        components are orthonormal and raw MSE already equals reconstruction error by
        Parseval's theorem); p=0 gives plain normalized MSE (full 1/std^2 inverse-variance
        penalty); p=1 (default) is the midpoint, a 1/std penalty on raw error.
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
        # per-component loss weight: K * std_k^p / sum(std_j^p), rescaled so the average weight
        # is 1 (keeps the overall loss scale comparable to an unweighted baseline, rather than
        # shrinking by ~K from the sum-to-1 normalization on top of WeightedL2Loss's mean
        # reduction). See the weight_power docstring above for what p controls.
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
    def load(cls, filepath, lazy=True, n_components=None, device=None, reflect_x=False, pmt_flip_perm=None):
        return cls(filepath, lazy=lazy, n_components=n_components, device=device, reflect_x=reflect_x, pmt_flip_perm=pmt_flip_perm)

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

    def contain(self, pos):
        """Check containment in the (possibly full symmetric) volume."""
        if self._reflect_x:
            return self.full_meta.contain(pos)
        return self.meta.contain(pos)

    def clear_prefetch(self):
        """Clear event-level prefetch cache."""
        self._event_uniq = None
        self._event_vis = None
        self._event_t0 = None
        self._event_coeffs = None

    def prefetch_event(self, positions, device=None):
        """Prefetch all unique voxels for this event.
        positions: (N, 3). device: where to store cached tensors (defaults to self._device).
        Call clear_prefetch() or prefetch_event(None) to clear."""
        if positions is None:
            self.clear_prefetch()
            return
        dev = device if device is not None else self._device
        pos = np.atleast_2d(np.asarray(positions, dtype=np.float64))
        flip = pos[:, 0] > 0 if self._reflect_x else np.zeros(len(pos), dtype=bool)
        pos_q = pos.copy()
        pos_q[flip, 0] *= -1
        vox = self.meta.coord_to_voxel(pos_q).numpy()
        event_uniq, _ = np.unique(vox, return_inverse=True)
        if event_uniq.size == 0:
            self._event_uniq = np.array([], dtype=np.int64)
            self._event_vis = torch.zeros(0, self._n_pmts, dtype=torch.float32, device=dev)
            t0_dtype = self.t0.dtype if self.t0 is not None else torch.long
            self._event_t0 = torch.zeros(0, self._n_pmts, dtype=t0_dtype, device=dev)
            self._event_coeffs = torch.zeros(0, self._n_pmts, self._n_components, dtype=torch.float32, device=dev)
            return
        data = self[event_uniq]
        self._event_uniq = event_uniq
        self._event_vis = data["vis"].to(dev)
        self._event_t0 = data["t0"].to(dev)
        self._event_coeffs = data["coeffs"].to(dev)

    def lookup(self, pos):
        """pos: (N, 3) array or tensor -> dict with vis, t0, coeffs. Handles x-reflection and PMT permutation."""
        pos = np.atleast_2d(np.asarray(pos, dtype=np.float64))
        flip = pos[:, 0] > 0 if self._reflect_x else np.zeros(len(pos), dtype=bool)
        pos_q = pos.copy()
        pos_q[flip, 0] *= -1
        vox = self.meta.coord_to_voxel(pos_q).numpy()

        if self._event_uniq is not None:
            inv_b = np.searchsorted(self._event_uniq, vox)
            dev = self._event_vis.device
            inv_t = torch.from_numpy(inv_b).long().to(dev)
            data = {
                "vis": self._event_vis[inv_t].clone(),
                "t0": self._event_t0[inv_t].clone(),
                "coeffs": self._event_coeffs[inv_t].clone(),
            }
        else:
            if self._file is None:
                vox = torch.from_numpy(vox).long().to(self.vis.device)
            data = self[vox]

        if np.any(flip):
            perm = torch.from_numpy(self._pmt_flip_perm).to(data["vis"].device)
            flip_t = torch.from_numpy(flip).to(data["vis"].device)
            data["vis"] = data["vis"].clone()
            data["t0"] = data["t0"].clone()
            data["coeffs"] = data["coeffs"].clone()
            data["vis"][flip_t] = data["vis"][flip_t][:, perm]
            data["t0"][flip_t] = data["t0"][flip_t][:, perm]
            data["coeffs"][flip_t] = data["coeffs"][flip_t][:, perm, :]
        return data

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

    def reconstruct_waveform(self, vis, coeffs, t0, n_bins=1000):
        """Unnormalized waveform: vis * pdf. (..., 81), (..., 81, K), (..., 81) -> (..., 81, n_bins)."""
        cdf = self.reconstruct_cdf(coeffs, t0, n_bins=n_bins)
        pdf = torch.diff(cdf, dim=-1, prepend=torch.zeros_like(cdf[..., :1]))
        vis = vis.to(cdf.device)
        if vis.dim() == cdf.dim() - 1:
            vis = vis.unsqueeze(-1)
        return vis * pdf

@DATASETS.register_module()
class CompressedPLibDataset(Dataset):
    """Map-style dataset for CompressedPLib. Returns position + target (vis, log_vis, t0, coeffs)."""

    def __init__(self, cfg, rank=0, world_size=1):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        plib_cfg = cfg.get("compressed_plib", cfg.get("photonlib", {}))
        filepath = plib_cfg.get("filepath")
        lazy = plib_cfg.get("lazy", True)
        n_components = plib_cfg.get("n_components")
        vis_eps = plib_cfg.get("vis_eps", 1e-6)
        self._vis_eps = vis_eps
        self._n_photon = float(plib_cfg.get("n_photon", 1.0))

        # Device option: "cuda" to preload everything on GPU
        self._device = torch.device(plib_cfg.get("device", "cpu"))
        self._on_gpu = self._device.type == "cuda"

        self._plib = CompressedPLib.load(filepath, lazy=(lazy and not self._on_gpu), n_components=n_components)

        self._normalize_coeffs = bool(plib_cfg.get("normalize_coeffs", False))
        if self._normalize_coeffs:
            # NOTE: every compressed_plib config in this repo sets this as "weight_power" (see
            # e.g. train_sirentv_81_dualpca_grad_supervision.yaml) -- this used to read
            # "coeff_weight_power" instead, a key that was never actually set anywhere, so this
            # silently defaulted to 1 in every run to date regardless of the configured value.
            coeff_weight_power = plib_cfg.get("weight_power", 1)
            print("[CompressedPLibDataset] computing per-component coeff stats for normalization "
                  f"(weight_power={coeff_weight_power}) ...")
            self._plib.compute_coeff_stats(weight_power=coeff_weight_power)
            print(f"  coeff_std range: [{self._plib.coeff_std.min():.4g}, {self._plib.coeff_std.max():.4g}]")
            print(f"  coeff_weight range: [{self._plib.coeff_weight.min():.4g}, {self._plib.coeff_weight.max():.4g}]")

        xform_cfg = cfg.get("transform_vis", {})
        self._xform_vis, self._inv_xform_vis = partial_xform_vis(xform_cfg)

        data_cfg = cfg.get("data", {})
        total_voxels = len(self._plib)
        max_len = data_cfg.get("max_len", -1)
        if max_len is None or max_len < 0:
            max_len = total_voxels
        effective = min(total_voxels, max_len)
        subsample_factor = data_cfg.get("subsample_factor", 1)
        if subsample_factor > 1:
            all_idx = torch.arange(effective, dtype=torch.long)
            indices = all_idx[::subsample_factor]
        else:
            indices = torch.arange(effective, dtype=torch.long)

        self.indices = indices

        # Target-side spatial-gradient precomputation (Frobenius norm across channels) for
        # whichever keys train.grad_supervision_keys asks for -- independent of whether the
        # main dataset path below is lazy, since a finite-difference gradient inherently needs
        # a full pass over the data regardless (same one-time-cost reasoning as
        # compute_coeff_stats above).
        self._grad_keys = [
            k for k in cfg.get("train", {}).get("grad_supervision_keys", []) if k in ("v", "coeffs")
        ]
        # EXACT (native-resolution, no coarse-graining) gradient-target precomputation via a
        # chunked Hutchinson random-projection read (see compute_grad_frob_target_exact_projected)
        # -- replaces the earlier subsample_factor + near-wall-refine two-tier approximation
        # (which had a real, confirmed systematic truncation-error bias at subsample_factor>1
        # that ensemble-averaging couldn't fix) with an exact computation that never needs to
        # avoid reading the full volume in the first place, since it never materializes the
        # full (N, n_pmt, K) array regardless of K.
        # number of independent Rademacher projections to average over for multi-channel keys
        # ("coeffs"); "v" (K=1) is always exact regardless (a single projection of a K=1 field
        # is exact, see compute_grad_frob_target_exact_projected's docstring).
        self._grad_n_projections = int(cfg.get("train", {}).get("grad_target_n_projections", 10))
        # voxels per chunked read (see compute_grad_frob_target_exact_projected) -- keeps at
        # most one chunk's raw K-dimensional slice in memory at once. null = no chunking.
        self._grad_chunk_size = cfg.get("train", {}).get("grad_target_chunk_size", 200000)
        # must match train.grad_supervision_aggregate (see SirenTV.forward's grad_aggregate) --
        # the prediction side aggregates all PMTs into one per-voxel value weighted by its own
        # (predicted) visibility, so the target side needs the matching (N,) shape, aggregated
        # by the TRUE visibility (the one thing the target side has that the prediction side
        # doesn't -- a real, accepted asymmetry, see the comment in model.py's forward()).
        self._grad_aggregate = bool(cfg.get("train", {}).get("grad_supervision_aggregate", False))
        # chunk size for the aggregation step's "vis" read (see _precompute_grad_targets) --
        # null/unset means no chunking (read the full array in one shot).
        self._grad_agg_chunk_size = cfg.get("train", {}).get("grad_target_agg_chunk_size", 20000)
        if self._grad_keys:
            self._precompute_grad_targets()

        # Precompute all targets on device (GPU or CPU)
        if self._on_gpu or not lazy:
            self._precompute_targets()

    def _grad_cache_meta(self):
        """Meta dict identifying exactly which config choices affect the cached per-(voxel,
        PMT) grad-frob targets -- anything here changing must invalidate a cache computed under
        the old value. See load_grad_frob_cache/save_grad_frob_cache in sirentv.utils.misc."""
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
            "n_projections": self._grad_n_projections,
            "seed": 0,
        }

    def _precompute_grad_targets(self):
        """One-time finite-difference gradient-target precomputation, cached in self._grad_targets
        (dataset-local order, matching self.indices) for whichever keys are requested. Computed
        EXACTLY at native grid resolution via a chunked Hutchinson random-projection read (see
        compute_grad_frob_target_exact_projected) -- reads the full volume in chunks, but never
        materializes more than one chunk's raw (chunk, n_pmt, K) slice in memory at once,
        regardless of K.

        The expensive per-(voxel, PMT) part (before any visibility-weighted aggregation) is
        cached to train.grad_target_cache_file if set, keyed on the exact config choices that
        affect it (see _grad_cache_meta) -- computed/cached over ALL voxels of the source LUT
        regardless of this run's own self.indices, so one cache file is reusable across
        differently-subsampled runs. See sirentv/scripts/precompute_grad_frob_targets.py to
        populate this once, offline (optionally on GPU via train.grad_target_device), instead of
        paying this cost at the start of every training/eval run."""
        from sirentv.utils.misc import (
            compute_grad_frob_target_exact_projected, load_grad_frob_cache, save_grad_frob_cache,
        )

        idx = self.indices.numpy()
        n_pmt = self._plib._n_pmts
        total_voxels = len(self._plib)
        cache_file = self.cfg.get("train", {}).get("grad_target_cache_file")
        meta = self._grad_cache_meta()

        cached = load_grad_frob_cache(cache_file, self._grad_keys, meta)
        if cached is not None:
            print(f"[CompressedPLibDataset] loaded EXACT target spatial-gradient for "
                  f"{self._grad_keys} from cache: {cache_file}")
            full_targets = cached
        else:
            # compute over ALL voxels (not just self.indices) so the cache stays valid for any
            # future run's own subsampling -- see _grad_cache_meta's docstring
            full_idx = np.arange(total_voxels)
            pos_np = self._plib.pos[full_idx]

            # TWO different validity criteria, deliberately not the same mask:
            # - valid_per_pmt (used for "coeffs"): a (voxel, PMT) pair with zero visibility to THAT
            #   SPECIFIC PMT has no real waveform, so its coeffs can be NaN or otherwise meaningless
            #   -- must be skipped as a NEIGHBOR in the finite difference, or it corrupts an
            #   adjacent real voxel's derivative. This is fundamentally per-(voxel, PMT): a voxel
            #   can be valid for one PMT and invalid for another (occluded, out of view, etc.).
            # - valid_agg (used for "v"): visibility=0 to one specific PMT is a real, physically
            #   meaningful value there (a shadowed/occluded region, not missing data) -- using
            #   valid_per_pmt for "v" would wrongly treat most of the volume as "invalid" for
            #   whichever PMT doesn't see it, degrading its gradient almost everywhere. "v" only
            #   needs the much rarer, genuinely-undefined case: zero visibility to EVERY PMT at
            #   once (inside a PMT's own solid body).
            # Both computed once here, chunked for the same reason the aggregation step below
            # chunks its "vis" read.
            n = total_voxels
            chunk_size = self._grad_chunk_size if self._grad_chunk_size is not None else n
            valid_per_pmt = np.empty((n, n_pmt), dtype=bool)
            for s in range(0, n, chunk_size):
                e = min(s + chunk_size, n)
                vis_chunk = self._plib[full_idx[s:e]]["vis"].numpy()
                valid_per_pmt[s:e] = vis_chunk > 0
            valid_agg = valid_per_pmt.any(axis=1)

            def read_v(chunk_idx):
                d = self._plib[full_idx[chunk_idx]]
                vis_norm = d["vis"].float() / self._n_photon
                return self._xform_vis(vis_norm).numpy()[:, :, None]  # (chunk, n_pmt, 1)

            def read_coeffs(chunk_idx):
                d = self._plib[full_idx[chunk_idx]]
                coeffs = d["coeffs"].float()
                if self._normalize_coeffs:
                    coeffs = self._plib.normalize_coeffs(coeffs)
                # defense-in-depth: valid_mask (above) is what actually prevents these voxels'
                # values from corrupting a neighbor's derivative; this just keeps any NaN here from
                # also leaking into diagnostics/logging that don't go through the masked path.
                coeffs = torch.nan_to_num(coeffs, nan=0.0)
                return coeffs.numpy()  # (chunk, n_pmt, K)

            grad_target_device = self.cfg.get("train", {}).get("grad_target_device")
            # caps the (N, n_pmt, batch) projected-array memory regardless of n_projections --
            # see compute_grad_frob_target_exact_projected's projection_batch_size docstring for
            # why this matters (n_projections=100 without batching is a ~52GB array for the
            # full dualpca LUT and silently OOM-kills the process). Default 10 matches the
            # n_projections value this was already known to run safely at.
            grad_target_projection_batch_size = self.cfg.get("train", {}).get("grad_target_projection_batch_size", 10)
            full_targets = {}
            if "v" in self._grad_keys:
                print("[CompressedPLibDataset] computing EXACT target spatial-gradient (v) ...")
                full_targets["v"] = compute_grad_frob_target_exact_projected(
                    pos_np, read_v, n_pmt=n_pmt, n_channels=1, n_projections=1,
                    chunk_size=self._grad_chunk_size, valid=valid_agg, device=grad_target_device,
                )
            if "coeffs" in self._grad_keys:
                print(f"[CompressedPLibDataset] computing EXACT target spatial-gradient (coeffs) "
                      f"via {self._grad_n_projections} random projections ...")
                full_targets["coeffs"] = compute_grad_frob_target_exact_projected(
                    pos_np, read_coeffs, n_pmt=n_pmt, n_channels=self._plib._n_components,
                    n_projections=self._grad_n_projections, chunk_size=self._grad_chunk_size,
                    valid=valid_per_pmt, device=grad_target_device,
                    projection_batch_size=grad_target_projection_batch_size,
                )

            if cache_file:
                print(f"[CompressedPLibDataset] caching EXACT target spatial-gradient for "
                      f"{self._grad_keys} to {cache_file}")
                save_grad_frob_cache(cache_file, full_targets, meta)

        self._grad_targets = {k: torch.from_numpy(np.asarray(v)[idx]).float() for k, v in full_targets.items()}

        if self._grad_aggregate:
            print("[CompressedPLibDataset] aggregating gradient targets across PMTs "
                  "(visibility-weighted) ...")
            # chunked, like compute_coeff_stats -- a full-array self._plib[idx]["vis"] read
            # (~500MB-1GB for ~1.6M voxels x 81 pmts) was enough to OOM here even though "vis"
            # is the smallest field in this file, so the available headroom is evidently much
            # tighter than that; never materialize more than one chunk's worth at a time.
            # grad_target_agg_chunk_size: null/unset -> no chunking (one shot, old behavior).
            n = len(idx)
            chunk_size = self._grad_agg_chunk_size if self._grad_agg_chunk_size is not None else n
            aggregated = {key: torch.empty(n, dtype=torch.float32) for key in self._grad_targets}
            for s in range(0, n, chunk_size):
                e = min(s + chunk_size, n)
                vis_chunk = self._plib[idx[s:e]]["vis"].float() / self._n_photon  # (chunk, n_pmt)
                for key, per_pmt in self._grad_targets.items():
                    # per_pmt is already the SQUARED per-(voxel, PMT) Frobenius norm (see
                    # compute_grad_frob_target's docstring) -- aggregating across PMTs is then
                    # just the visibility-weighted SUM, no re-squaring and no final sqrt, to
                    # match compute_grad_frob_hutchinson_aggregate's own squared-domain
                    # convention on the prediction side (see loss/grad_frob.py).
                    aggregated[key][s:e] = (vis_chunk * per_pmt[s:e]).sum(dim=1)
            self._grad_targets = aggregated

    def _precompute_targets(self):
        """Precompute and store all transformed targets on self._device."""
        idx = self.indices.numpy()
        pos_np = self._plib.pos[idx]
        self._pos = torch.from_numpy(pos_np).float().to(self._device)

        data = self._plib[idx]
        vis_raw = data["vis"].float()
        t0_raw = data["t0"].float()
        coeffs = data["coeffs"].float()

        vis_norm = vis_raw / self._n_photon
        v = self._xform_vis(vis_norm)
        t0 = torch.log(t0_raw.clamp(min=1e-3))

        if self._normalize_coeffs:
            coeffs = self._plib.normalize_coeffs(coeffs)

        self._v = v.to(self._device)
        self._t0 = t0.to(self._device)
        self._coeffs = coeffs.to(self._device)
        self._vis_raw = vis_raw.to(self._device)
        self._t0_raw = t0_raw.to(self._device)

        # Free the underlying plib data
        self._plib.close()
        print(f"[CompressedPLibDataset] precomputed {len(self.indices)} voxels on {self._device}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if hasattr(self, "_v"):
            # Precomputed path (GPU or eager CPU)
            target = {
                "v": self._v[idx],
                "t0": self._t0[idx],
                "coeffs": self._coeffs[idx],
            }
            if self._normalize_coeffs:
                # per-component loss weight (see compute_coeff_stats) -- (1, K), same for every
                # sample, broadcasts against coeffs' (n_pmts, K) shape once collated to (B, 1, K)
                target["coeffs_weight"] = self._plib.coeff_weight.unsqueeze(0).cpu()
            for key in self._grad_keys:
                target[f"{key}_grad_frob"] = self._grad_targets[key][idx]
            return {
                "position": self._pos[idx],
                "target": target,
                "meta": {
                    "v_linear": self._vis_raw[idx],
                    "t0_raw": self._t0_raw[idx],
                    "voxel_id": self.indices[idx],
                },
            }

        # Lazy CPU path
        vox_id = self.indices[idx].item()
        pos = torch.from_numpy(self._plib.pos[vox_id]).float()
        data = self._plib[vox_id]
        vis_raw = data["vis"].float()
        t0_raw = data["t0"].float()
        coeffs = data["coeffs"].float()

        if self._normalize_coeffs:
            coeffs = self._plib.normalize_coeffs(coeffs)

        vis_norm = vis_raw / self._n_photon
        v = self._xform_vis(vis_norm)
        t0 = torch.log(t0_raw.clamp(min=1e-3))

        target = {
            "v": v,
            "t0": t0,
            "coeffs": coeffs,
        }
        if self._normalize_coeffs:
            target["coeffs_weight"] = self._plib.coeff_weight.unsqueeze(0).cpu()
        for key in self._grad_keys:
            target[f"{key}_grad_frob"] = self._grad_targets[key][idx]

        return {
            "position": pos,
            "target": target,
            "meta": {
                "v_linear": vis_raw,
                "t0_raw": t0_raw,
                "voxel_id": self.indices[idx],
            },
        }


def create_compressed_dataloader(cfg, rank=0, world_size=1):
    """Create DataLoader for CompressedPLibDataset."""
    dataset = CompressedPLibDataset(cfg, rank=rank, world_size=world_size)
    loader_cfg = cfg.get("data", {}).get("loader", {})
    batch_size = loader_cfg.get("batch_size", 1)
    num_workers = loader_cfg.get("num_workers", 0)
    pin_memory = loader_cfg.get("pin_memory", False)
    drop_last = loader_cfg.get("drop_last", True)
    shuffle = loader_cfg.get("shuffle", False)

    # GPU data requires num_workers=0 (workers can't access GPU tensors)
    if dataset._on_gpu:
        if num_workers > 0:
            print("[create_compressed_dataloader] forcing num_workers=0 for GPU data")
        num_workers = 0
        pin_memory = False
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle = False
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=True if num_workers > 0 else False,
    )
    if sampler is not None:
        dl.set_epoch = lambda e: sampler.set_epoch(e)
    return dl