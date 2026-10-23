"""
Compressed photon library reader and dataset for (vis, t0, quantiles) format.
"""

import os
import time

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from slar.transform import partial_xform_vis
from photonlib.meta import VoxelMeta

from sirentv.data.builder import DATASETS

class QuantilePLib:
    """
    Reader for quantile waveform photon library H5 (vis, t0, quantiles).
    """

    def __init__(self, filepath, lazy=True, mode="quantile", combine_every_quantile=1, device=None, reflect_x=False, pmt_flip_perm=None, debug_timing=False, log_quantile_C=None):
        self._path = filepath
        self._lazy = lazy
        self._device = device or torch.device("cpu")
        self._file = None
        self._reflect_x = bool(reflect_x)
        self._mode = mode
        self._debug_timing = debug_timing

        with h5py.File(filepath, "r") as f:
            self._n_voxels = f["vis"].shape[0]
            self._n_pmts = f["vis"].shape[1]
            self._n_quantiles = f["quantiles"].shape[2]
            self._combine_every_quantile = combine_every_quantile
            self._t0_in_ns = (
                f.attrs.get("t0_in_ns", False) or self._mode == "quantile"
                or f["analytical_t0"].dtype.kind == "f"
            )
            self._t_max_ns = f.attrs.get("t_max_ns", 600.0)
            self._n_bins = f.attrs.get("n_bins", 1000)

            if log_quantile_C is not None:
                self._log_quantile_C = float(log_quantile_C)
            else:
                self._log_quantile_C = f.attrs.get("log_quantile_C", 1e-2)

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


        self._event_uniq = None
        self._event_vis = None
        self._event_t0 = None
        self._event_quantiles = None

        if lazy:
            self._file = h5py.File(filepath, "r", swmr=True, libver="latest")
            self._vis = self._file["vis"]
            self._t0 = self._file["analytical_t0"]
            self._quantiles = self._file["quantiles"]
            self.vis = None
            self.t0 = None
            self.quantiles = None
        else:
            self._file = None
            with h5py.File(filepath, "r") as f:
                self.vis = torch.from_numpy(f["vis"][:]).float()
                t0_arr = f["analytical_t0"][:]
                if self._t0_in_ns:
                    self.t0 = torch.from_numpy(t0_arr).float()
                else:
                    self.t0 = torch.from_numpy(t0_arr).long().clamp(min=0)
                
                assert self._n_quantiles % self._combine_every_quantile == 0, "n_quantiles must be divisible by combine_every_quantile"
                q_part = torch.from_numpy(f["quantiles"][:, :, : self._n_quantiles])
                self.quantiles = q_part[:, :, ::self._combine_every_quantile].float().to(self._device)
            self._vis = self._t0 = self._quantiles = None

    def __len__(self):
        return self._n_voxels


    @classmethod
    def load(cls, filepath, lazy=True, mode="quantile", combine_every_quantile=1, device=None, reflect_x=False, pmt_flip_perm=None, debug_timing=False, log_quantile_C=None):
        return cls(filepath, lazy=lazy, mode=mode, combine_every_quantile=combine_every_quantile, device=device, reflect_x=reflect_x, pmt_flip_perm=pmt_flip_perm, debug_timing=debug_timing, log_quantile_C=log_quantile_C)

    def read_strided(self, start, stop, step, fields=None):
        """Read rows start:stop:step of the requested fields as ONE hyperslab per field.

        For loading a whole rank shard up front: a strided slice is a single HDF5 selection,
        where the fancy-index path in _read_slice issues a point selection that is far slower
        for tens of thousands of rows. Requires the lazy (file-backed) reader.
        """
        if self._file is None:
            raise RuntimeError("read_strided needs a lazy QuantilePLib (file-backed)")
        fields = self.FIELDS if fields is None else tuple(fields)
        out = {}
        for f in fields:
            arr = getattr(self, f"_{f}")[start:stop:step]
            if f == "quantiles":
                n = self._n_quantiles // self._combine_every_quantile
                arr = np.ascontiguousarray(arr[..., ::self._combine_every_quantile][..., :n])
            out[f] = torch.from_numpy(np.asarray(arr))
        return out

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
        
        if key == "quantiles":
            arr = arr[..., ::self._combine_every_quantile]
        return torch.from_numpy(np.asarray(arr)).to(self._device)

    FIELDS = ("vis", "t0", "quantiles")

    def __getitem__(self, voxel_ids, fields=None):
        """Read the requested fields for these voxel(s). Default: all of FIELDS.

        `fields` exists because a single-target run consumes only one of them, and the
        payloads are wildly asymmetric: one voxel of `quantiles` is 81 x 512 x 4 B = 162 KiB
        against ~0.3 KiB for `vis` or `t0`. Reading all three for a v-only or t0-only branch
        therefore costs ~500x the bandwidth it uses and makes those runs data-bound on data
        they discard. QuantilePLibDataset derives the set from which targets it was asked
        for -- see _fields_for_targets.
        """
        fields = self.FIELDS if fields is None else tuple(fields)
        unknown = [f for f in fields if f not in self.FIELDS]
        if unknown:
            raise ValueError(f"unknown field(s) {unknown}; expected a subset of {self.FIELDS}")

        out = {}
        if self._file is None and not isinstance(voxel_ids, (int, np.integer)):
            vox = voxel_ids
            if isinstance(vox, np.ndarray):
                vox = torch.from_numpy(vox).long().to(self.vis.device)
            uniq, inv = torch.unique(vox, return_inverse=True)
            for f in fields:
                out[f] = torch.index_select(getattr(self, f), 0, uniq)[inv].clone()
        else:

            for f in fields:
                out[f] = self._read_slice(f, voxel_ids)

        if "quantiles" in out:
            q = out["quantiles"]
            n = self._n_quantiles // self._combine_every_quantile
            out["quantiles"] = q[:, :n] if q.dim() == 2 else q[..., :n]
        return out

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
        self._event_quantiles = None

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
            self._event_quantiles = torch.zeros(0, self._n_pmts, self._n_quantiles//self._combine_every_quantile, dtype=torch.float32, device=dev)
            return
        data = self[event_uniq]
        self._event_uniq = event_uniq
        self._event_vis = data["vis"].to(dev)
        self._event_t0 = data["t0"].to(dev)
        self._event_quantiles = data["quantiles"].to(dev)

    def lookup(self, pos):
        """pos: (N, 3) array or tensor -> dict with vis, t0, quantiles. Handles x-reflection and PMT permutation."""
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
                "vis_mask": self._event_vis_mask[inv_t].clone(),
                "t0": self._event_t0[inv_t].clone(),
                "quantiles": self._event_quantiles[inv_t].clone(),
            }
        else:
            if self._file is None:
                vox = torch.from_numpy(vox).long().to(self.vis.device)
            data = self[vox]

        if np.any(flip):
            perm = torch.from_numpy(self._pmt_flip_perm).to(data["vis"].device)
            flip_t = torch.from_numpy(flip).to(data["vis"].device)
            data["vis"] = data["vis"].clone()
            data["vis_mask"] = data["vis_mask"].clone()
            data["t0"] = data["t0"].clone()
            data["quantiles"] = data["quantiles"].clone()
            data["vis"][flip_t] = data["vis"][flip_t][:, perm]
            data["t0"][flip_t] = data["t0"][flip_t][:, perm]
            data["quantiles"][flip_t] = data["quantiles"][flip_t][:, perm, :]
        return data

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
       
    @property
    def mode(self):
        """Quantile mode: 'quantile' or 'log_quantile'."""
        return self._mode

    def reconstruct_aligned_cdf(self, u_grid, quantiles, t0=None):
        """t0: (..., n_pmt) onset times in ns to add back for quantiles that were trained
        t0-aligned (photonlib.align_quantiles_to_t0), so the CDF is on the absolute time axis
        like the LUT's own. None leaves the quantiles as they are."""
        u_np = u_grid.astype(np.float32)
        q = quantiles.detach().cpu().numpy()

        assert u_grid.shape[-1] == quantiles.shape[-1], "u_grid and quantiles must have the same number of channels"

        if self._mode == "log_quantile":
            q = np.power(10.0, q) - self._log_quantile_C
        q = np.clip(q, 0.0, None)
        if t0 is not None:
            q = q + t0.detach().cpu().numpy()[..., None]
        

        batch_shape = q.shape[:-1]
        flat = q.reshape(-1, q.shape[-1])
        t_edges = np.linspace(0, self._t_max_ns, self._n_bins + 1, dtype=np.float32)
        out = np.stack([
            np.interp(t_edges, flat[i], u_np, left=0.0, right=1.0)[:-1]
            for i in range(flat.shape[0])
        ])
        return torch.from_numpy(out.reshape(*batch_shape, self._n_bins)).to(
            device=quantiles.device, dtype=quantiles.dtype
        )

    def to_linear_time(self, quantiles, t0=None):
        """quantiles: (..., n_quantile) raw (possibly log_quantile-transformed) quantile
        function values -> linear-ns quantile times (..., n_quantile). Unlike
        CompressedPLib.to_linear_time, there's no PCA reconstruction step here -- quantiles
        are already this library's native per-channel representation -- so this only inverts
        the log_quantile transform (same formula/clamp as CompressedPLib.to_linear_time).

        t0: (..., n_pmt) onset times in ns, added back after the inversion for quantiles that
        were trained t0-aligned (photonlib.align_quantiles_to_t0), giving absolute times again.
        """
        if self._mode == "log_quantile":
            q = torch.pow(10.0, quantiles) - self._log_quantile_C
        else:
            q = quantiles
        q = q.clamp(min=0.0)
        if t0 is not None:
            q = q + t0.unsqueeze(-1)
        return q


@DATASETS.register_module()
class QuantilePLibDataset(Dataset):
    """Map-style dataset for QuantilePLib. Returns position + target (vis, log_vis, t0, quantiles)."""

    def __init__(self, cfg, rank=0, world_size=1):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        plib_cfg = cfg.get("quantile_plib", cfg.get("photonlib", {}))
        filepath = plib_cfg.get("filepath")
        lazy = plib_cfg.get("lazy", True)
        combine_every_quantile = plib_cfg.get("combine_every_quantile", 1)
        vis_eps = plib_cfg.get("vis_eps", 1e-6)
        self._vis_eps = vis_eps
        self._n_photon = float(plib_cfg.get("n_photon", 1.0))
        mode = plib_cfg.get("mode", "quantile")
        # Device option: "cuda" to preload everything on GPU
        self._device = torch.device(plib_cfg.get("device", "cpu"))
        self._on_gpu = self._device.type == "cuda"

        debug_timing = bool(cfg.get("debug_timing", False))
        self._quantile_plib = QuantilePLib.load(
            filepath, lazy=(lazy and not self._on_gpu), mode=mode,
            combine_every_quantile=combine_every_quantile, debug_timing=debug_timing,
            log_quantile_C=plib_cfg.get("log_quantile_C"),
        )
        if mode == "log_quantile":
            self._log_quantile = True
            self._log_quantile_C = self._quantile_plib._log_quantile_C
        else:
            self._log_quantile = False

        self._align_t0 = bool(plib_cfg.get("align_quantiles_to_t0", False))

        xform_cfg = cfg.get("transform_vis", {})
        self._xform_vis, self._inv_xform_vis = partial_xform_vis(xform_cfg)

        data_cfg = cfg.get("data", {})
        total_voxels = len(self._quantile_plib)
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

        self._target_keys = data_cfg.get("target_keys")
        if self._target_keys is not None:
            self._target_keys = list(self._target_keys)
            known = ("v", "t0", "quantiles")
            unknown = [k for k in self._target_keys if k not in known]
            if unknown:
                raise ValueError(
                    f"data.target_keys {unknown} unknown; expected a subset of {known}"
                )
        self._read_fields = self._fields_for_targets(self._target_keys, self._align_t0)
        if self._target_keys is not None and rank == 0:
            print(f"[QuantilePLibDataset] targets {self._target_keys} "
                  f"-> reading LUT fields {sorted(self._read_fields)}")


        self._grad_keys = [
            k for k in cfg.get("train", {}).get("grad_supervision_keys", []) if k in ("v", "quantiles")
        ]

        self._grad_n_projections = int(cfg.get("train", {}).get("grad_target_n_projections", 10))
        self._grad_chunk_size = cfg.get("train", {}).get("grad_target_chunk_size", 200000)
        self._grad_aggregate = bool(cfg.get("train", {}).get("grad_supervision_aggregate", False))
        self._grad_agg_chunk_size = cfg.get("train", {}).get("grad_target_agg_chunk_size", 20000)
        if self._grad_keys:
            self._precompute_grad_targets()

        # data.memory_shard: hold this rank's 1/world_size of the voxels resident in memory
        # and iterate it with MemoryShardLoader, instead of random HDF5 reads every batch. The
        # shard is the strided slice indices[rank::world_size], so every rank sees the whole
        # detector, and each epoch still visits every voxel exactly once -- but a voxel stays
        # on the same rank for the whole run, where DistributedSampler re-deals the voxels
        # across ranks every epoch.

        self._memory_shard = bool(data_cfg.get("memory_shard", False))
        if self._memory_shard:
            if self._on_gpu or not lazy:
                raise ValueError(
                    "data.memory_shard loads the shard itself; keep photonlib.lazy: true and "
                    "photonlib.device: cpu (the eager paths would load the WHOLE LUT per rank)"
                )
            n_per_rank = len(self.indices) // world_size
            self.indices = self.indices[rank::world_size][:n_per_rank]

            self._grad_targets = {
                k: v[rank::world_size][:n_per_rank] for k, v in getattr(self, "_grad_targets", {}).items()
            }
            shard_device = data_cfg.get("memory_shard_device", "cpu")
            if shard_device == "cuda":
                self._device = torch.device("cuda", torch.cuda.current_device())
            elif shard_device == "cpu":
                self._device = torch.device("cpu")
            else:
                raise ValueError(f"data.memory_shard_device must be 'cpu' or 'cuda', got {shard_device!r}")
            self._on_gpu = self._device.type == "cuda"
            self._precompute_shard(int(data_cfg.get("memory_shard_chunk", 1024)))
        # Precompute all targets on device (GPU or CPU)
        elif self._on_gpu or not lazy:
            self._precompute_targets()

    def _grad_cache_meta(self):
        """Meta dict identifying exactly which config choices affect the cached per-(voxel,
        PMT) grad-frob targets -- anything here changing must invalidate a cache computed under
        the old value. See load_grad_frob_cache/save_grad_frob_cache in sirentv.utils.misc."""
        from sirentv.utils.misc import GRAD_CACHE_FORMAT_VERSION

        xform_cfg = self.cfg.get("transform_vis", {})
        plib_cfg = self.cfg.get("quantile_plib", self.cfg.get("photonlib", {}))
        # Added only when set, so caches computed before these options existed (absolute
        # quantiles, the file's C) still match their unchanged configs.
        extra = {}
        if self._align_t0:
            extra["align_quantiles_to_t0"] = True
        if plib_cfg.get("log_quantile_C") is not None:
            extra["log_quantile_C"] = float(plib_cfg["log_quantile_C"])
        return {
            **extra,
            "format_version": GRAD_CACHE_FORMAT_VERSION,
            "lut_filepath": plib_cfg.get("filepath"),
            "total_voxels": len(self._quantile_plib),
            "n_photon": self._n_photon,
            "xform_vmax": xform_cfg.get("vmax", 1.0),
            "xform_eps": xform_cfg.get("eps", 1e-8),
            "xform_sin_out": xform_cfg.get("sin_out", False),
            "mode": plib_cfg.get("mode", "quantile"),
            "combine_every_quantile": plib_cfg.get("combine_every_quantile", 1),
            "n_projections": self._grad_n_projections,
            "seed": 0,
        }

    def _precompute_grad_targets(self):
        """One-time finite-difference gradient-target precomputation, cached in self._grad_targets
        (dataset-local order, matching self.indices) for whichever keys are requested. Computed
        EXACTLY at native grid resolution via a chunked Hutchinson random-projection read (see
        compute_grad_frob_target_exact_projected) -- reads the full volume in chunks, but never
        materializes more than one chunk's raw (chunk, n_pmt, K) slice in memory at once,
        regardless of K. Identical mechanism to CompressedPLibDataset._precompute_grad_targets,
        including the offline cache (train.grad_target_cache_file) -- see that method's
        docstring and sirentv/scripts/precompute_grad_frob_targets.py."""
        from sirentv.utils.misc import (
            compute_grad_frob_target_exact_projected, load_grad_frob_cache, save_grad_frob_cache,
        )

        idx = self.indices.numpy()
        n_pmt = self._quantile_plib._n_pmts
        n_quantile_channels = self._quantile_plib._n_quantiles // self._quantile_plib._combine_every_quantile
        total_voxels = len(self._quantile_plib)
        cache_file = self.cfg.get("train", {}).get("grad_target_cache_file")
        meta = self._grad_cache_meta()

        cached = load_grad_frob_cache(cache_file, self._grad_keys, meta)
        if cached is not None:
            print(f"[QuantilePLibDataset] loaded EXACT target spatial-gradient for "
                  f"{self._grad_keys} from cache: {cache_file}")
            full_targets = cached
        else:
            # compute over ALL voxels (not just self.indices) so the cache stays valid for any
            # future run's own subsampling -- see _grad_cache_meta's docstring
            full_idx = np.arange(total_voxels)
            pos_np = self._quantile_plib.pos[full_idx]

            n = total_voxels
            chunk_size = self._grad_chunk_size if self._grad_chunk_size is not None else n
            valid_per_pmt = np.empty((n, n_pmt), dtype=bool)
            for s in range(0, n, chunk_size):
                e = min(s + chunk_size, n)
                vis_chunk = self._quantile_plib[full_idx[s:e]]["vis"].numpy()
                valid_per_pmt[s:e] = vis_chunk > 0
            valid_agg = valid_per_pmt.any(axis=1)

            def read_v(chunk_idx):
                d = self._quantile_plib[full_idx[chunk_idx]]
                vis_norm = d["vis"].float() / self._n_photon
                return self._xform_vis(vis_norm).numpy()[:, :, None]  # (chunk, n_pmt, 1)

            def read_quantiles(chunk_idx):
                d = self._quantile_plib.__getitem__(full_idx[chunk_idx], fields=("quantiles", "t0"))
                quantiles = self._xform_quantiles(d["quantiles"].float(), d["t0"].float())
                # defense-in-depth: valid_mask (above) is what actually prevents these voxels'
                # values from corrupting a neighbor's derivative; this just keeps any NaN here from
                # also leaking into diagnostics/logging that don't go through the masked path.
                quantiles = torch.nan_to_num(quantiles, nan=0.0)
                return quantiles.numpy()  # (chunk, n_pmt, K)

            grad_target_device = self.cfg.get("train", {}).get("grad_target_device")
            # caps the (N, n_pmt, batch) projected-array memory regardless of n_projections --
            # see compute_grad_frob_target_exact_projected's projection_batch_size docstring for
            # why this matters (n_projections=100 without batching is tens of GB for the full
            # LUT and silently OOM-kills the process). Default 10 matches the n_projections
            # value this was already known to run safely at.
            grad_target_projection_batch_size = self.cfg.get("train", {}).get("grad_target_projection_batch_size", 10)
            full_targets = {}
            if "v" in self._grad_keys:
                print("[QuantilePLibDataset] computing EXACT target spatial-gradient (v) ...")
                full_targets["v"] = compute_grad_frob_target_exact_projected(
                    pos_np, read_v, n_pmt=n_pmt, n_channels=1, n_projections=1,
                    chunk_size=self._grad_chunk_size, valid=valid_agg, device=grad_target_device,
                )
            if "quantiles" in self._grad_keys:
                print(f"[QuantilePLibDataset] computing EXACT target spatial-gradient (quantiles) "
                      f"via {self._grad_n_projections} random projections ...")
                full_targets["quantiles"] = compute_grad_frob_target_exact_projected(
                    pos_np, read_quantiles, n_pmt=n_pmt, n_channels=n_quantile_channels,
                    n_projections=self._grad_n_projections, chunk_size=self._grad_chunk_size,
                    valid=valid_per_pmt, device=grad_target_device,
                    projection_batch_size=grad_target_projection_batch_size,
                )

            if cache_file:
                print(f"[QuantilePLibDataset] caching EXACT target spatial-gradient for "
                      f"{self._grad_keys} to {cache_file}")
                save_grad_frob_cache(cache_file, full_targets, meta)

        self._grad_targets = {k: torch.from_numpy(np.asarray(v)[idx]).float() for k, v in full_targets.items()}

        if self._grad_aggregate:
            print("[QuantilePLibDataset] aggregating gradient targets across PMTs "
                  "(visibility-weighted) ...")
            n = len(idx)
            chunk_size = self._grad_agg_chunk_size if self._grad_agg_chunk_size is not None else n
            aggregated = {key: torch.empty(n, dtype=torch.float32) for key in self._grad_targets}
            for s in range(0, n, chunk_size):
                e = min(s + chunk_size, n)
                vis_chunk = self._quantile_plib[idx[s:e]]["vis"].float() / self._n_photon  # (chunk, n_pmt)
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
        pos_np = self._quantile_plib.pos[idx]
        self._pos = torch.from_numpy(pos_np).float().to(self._device)

        data = self._quantile_plib[idx]
        vis_raw = data["vis"].float()
        t0_raw = data["t0"].float()
        quantiles = data["quantiles"].float()

        vis_norm = vis_raw / self._n_photon
        v = self._xform_vis(vis_norm)
        t0 = torch.log(t0_raw.clamp(min=1e-3))
        quantiles = self._xform_quantiles(quantiles, t0_raw)

        self._v = v.to(self._device)
        self._t0 = t0.to(self._device)
        self._quantiles = quantiles.to(self._device)
        self._vis_raw = vis_raw.to(self._device)
        self._t0_raw = t0_raw.to(self._device)

        # Free the underlying plib data
        self._quantile_plib.close()
        print(f"[CompressedPLibDataset] precomputed {len(self.indices)} voxels on {self._device}")

    def _xform_quantiles(self, quantiles, t0_raw):
        """Raw LUT quantiles (..., n_pmt, K) in ns -> the network's target domain.

        The ONE place the target transform lives, shared by the eager, lazy, shard and
        grad-target paths so they cannot drift apart. With align_quantiles_to_t0 the onset
        is removed first, so the log acts on the delay after t0, not on absolute time.
        """
        if self._align_t0:
            quantiles = quantiles - t0_raw.unsqueeze(-1)
        if self._log_quantile:
            quantiles = torch.log10(quantiles.clamp(min=0.0) + self._log_quantile_C)
        return quantiles

    def _precompute_shard(self, chunk_size):
        """Load this rank's strided shard into self._device memory, chunk by chunk.

        Reads rows directly through read_strided (one hyperslab per field per chunk) and
        writes each transformed chunk into preallocated storage, so the transient footprint
        is one chunk -- the eager _precompute_targets would hold several full-size copies
        (raw, float, transformed) at once, which for 4 ranks per node exceeds the node's RAM.
        """
        idx = self.indices.numpy()
        n = len(idx)
        n_pmt = self._quantile_plib._n_pmts
        n_q = self._quantile_plib._n_quantiles // self._quantile_plib._combine_every_quantile
        dev = self._device

        need = n * n_pmt * (n_q + 4) * 4 + n * 3 * 4
        if dev.type == "cuda":
            free, _ = torch.cuda.mem_get_info(dev)
            sharers = 1
        else:
            # every rank on the node loads its shard at the same time, out of the same RAM
            with open("/proc/meminfo") as f:
                meminfo = dict(line.split(":", 1) for line in f)
            free = int(meminfo["MemAvailable"].split()[0]) * 1024
            sharers = int(os.environ.get("SLURM_NTASKS_PER_NODE", "1").split("(")[0])
        if need * sharers > 0.8 * free:
            raise RuntimeError(
                f"data.memory_shard: {sharers} shard(s) of {need / 2**30:.1f} GiB need more than 80% of "
                f"the {free / 2**30:.1f} GiB available on {dev}; use more ranks, a different "
                f"memory_shard_device, or turn memory_shard off"
            )

        steps = np.diff(idx)
        strided = n > 0 and (n == 1 or np.all(steps == steps[0]))
        step = int(steps[0]) if n > 1 else 1

        self._pos = torch.from_numpy(self._quantile_plib.pos[idx]).float().to(dev)
        self._v = torch.empty(n, n_pmt, dtype=torch.float32, device=dev)
        self._t0 = torch.empty_like(self._v)
        self._vis_raw = torch.empty_like(self._v)
        self._t0_raw = torch.empty_like(self._v)
        self._quantiles = torch.empty(n, n_pmt, n_q, dtype=torch.float32, device=dev)

        t_start = time.time()
        n_neg = 0
        n_valid = 0
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            if strided:
                data = self._quantile_plib.read_strided(int(idx[s]), int(idx[e - 1]) + 1, step)
            else:
                data = self._quantile_plib[idx[s:e]]
            # .float() before .to(dev): vis is uint32 in the LUT, which CUDA barely supports
            vis_raw = data["vis"].float().to(dev)
            t0_raw = data["t0"].float().to(dev)
            quantiles = data["quantiles"].float().to(dev)
            if self._align_t0:
                # how often the first quantile falls BELOW analytical_t0 among visible pairs;
                # those delays are clamped to 0 by _xform_quantiles, so report the rate
                valid = vis_raw > 0
                n_neg += int(((quantiles[..., 0] - t0_raw) < 0)[valid].sum())
                n_valid += int(valid.sum())
            self._v[s:e] = self._xform_vis(vis_raw / self._n_photon)
            self._t0[s:e] = torch.log(t0_raw.clamp(min=1e-3))
            self._quantiles[s:e] = self._xform_quantiles(quantiles, t0_raw)
            self._vis_raw[s:e] = vis_raw
            self._t0_raw[s:e] = t0_raw
        self._grad_targets = {k: v.to(dev) for k, v in self._grad_targets.items()}

        self._quantile_plib.close()
        msg = (f"[QuantilePLibDataset] rank {self.rank}: memory_shard of {n} voxels "
               f"({need / 2**30:.1f} GiB) loaded onto {dev} in {time.time() - t_start:.0f} s")
        if self._align_t0:
            msg += f"; Q(u_min) < analytical_t0 for {n_neg}/{n_valid} visible (voxel, PMT) pairs"
        print(msg)

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def _fields_for_targets(target_keys, align_t0=False):
        """Which LUT fields must be read to build the requested targets.

        `quantiles` needs `vis` as well, not just `quantiles`: quantiles_mask is derived
        from the RAW target visibility (vis_raw > 0), because a zero-visibility (voxel, PMT)
        pair has no waveform and its quantiles are NaN in the LUT. Dropping vis there would
        silently feed those NaNs into the loss. With align_t0 it needs `t0` too, since the
        target is Q - analytical_t0 even for a quantile-only branch.
        """
        if target_keys is None:
            return set(QuantilePLib.FIELDS)
        need = {"v": {"vis"}, "t0": {"t0"}, "quantiles": {"quantiles", "vis"}}
        if align_t0:
            need["quantiles"] = {"quantiles", "vis", "t0"}
        fields = set()
        for k in target_keys:
            fields |= need[k]
        return fields

    def __getitem__(self, idx):
        if hasattr(self, "_v"):
            # Precomputed path (GPU or eager CPU)
            target = {
                "v": self._v[idx],
                "t0": self._t0[idx],
                "quantiles": self._quantiles[idx],
                "quantiles_mask": self._vis_raw[idx] > 0,
            }
            for key in self._grad_keys:
                target[f"{key}_grad_frob"] = self._grad_targets[key][idx]
            return {
                "position": self._pos[idx],
                "target": target,
                "meta": {
                    "v_linear": self._vis_raw[idx],
                    "t0_raw": self._t0_raw[idx],
                },
            }

        # Lazy CPU path
        vox_id = self.indices[idx].item()
        pos = torch.from_numpy(self._quantile_plib.pos[vox_id]).float()
        data = self._quantile_plib.__getitem__(vox_id, fields=self._read_fields)

        want = self._target_keys
        target, meta = {}, {}
        vis_raw = data["vis"].float() if "vis" in data else None
        if vis_raw is not None:
            meta["v_linear"] = vis_raw
        # t0 is read for a t0 target, and also for an aligned quantile target (see
        # _fields_for_targets); either way t0_raw goes into meta for consumers to add back
        t0_raw = data["t0"].float() if "t0" in data else None
        if t0_raw is not None:
            meta["t0_raw"] = t0_raw
        if want is None or "v" in want:
            target["v"] = self._xform_vis(vis_raw / self._n_photon)
        if want is None or "t0" in want:
            target["t0"] = torch.log(t0_raw.clamp(min=1e-3))
        if want is None or "quantiles" in want:
            target["quantiles"] = self._xform_quantiles(data["quantiles"].float(), t0_raw)
            target["quantiles_mask"] = vis_raw > 0
        for key in self._grad_keys:
            target[f"{key}_grad_frob"] = self._grad_targets[key][idx]

        return {"position": pos, "target": target, "meta": meta}

    def __getitems__(self, indices):
        """Fetch a whole batch with ONE read per field instead of one read per sample.

        See CompressedPLibDataset.__getitems__ for the mechanism and the measurements. The
        payoff is larger here: one voxel of `quantiles` is 162 KiB, and the per-sample path
        reaches only ~79 MB/s against ~1.2 GB/s for a single multi-row selection through the
        same three-level VDS chain.
        """
        if hasattr(self, "_v"):
            # Precomputed path: already resident, nothing to coalesce.
            return [self[i] for i in indices]

        idx = np.asarray(indices)
        vox = self.indices[torch.as_tensor(idx, dtype=torch.long)].numpy()
        data = self._quantile_plib.__getitem__(vox, fields=self._read_fields)

        want = self._target_keys
        cols, metacols = {}, {}
        vis_raw = data["vis"].float() if "vis" in data else None      # (B, n_pmt)
        if vis_raw is not None:
            metacols["v_linear"] = vis_raw
        t0_raw = data["t0"].float() if "t0" in data else None         # (B, n_pmt)
        if t0_raw is not None:
            metacols["t0_raw"] = t0_raw
        if want is None or "v" in want:
            cols["v"] = self._xform_vis(vis_raw / self._n_photon)
        if want is None or "t0" in want:
            cols["t0"] = torch.log(t0_raw.clamp(min=1e-3))
        if want is None or "quantiles" in want:
            # (B, n_pmt, n_quantile)
            cols["quantiles"] = self._xform_quantiles(data["quantiles"].float(), t0_raw)
            # Zero-visibility (voxel, PMT) pairs have NaN quantiles in the LUT; the mask is
            # what keeps them out of the loss. From the RAW visibility, as in __getitem__.
            cols["quantiles_mask"] = vis_raw > 0
        pos = torch.from_numpy(self._quantile_plib.pos[vox]).float()
        grads = {k: self._grad_targets[k][idx] for k in self._grad_keys}

        out = []
        for j in range(idx.shape[0]):
            target = {k: v[j] for k, v in cols.items()}
            for k in self._grad_keys:
                target[f"{k}_grad_frob"] = grads[k][j]
            out.append({
                "position": pos[j],
                "target": target,
                "meta": {k: v[j] for k, v in metacols.items()},
            })
        return out


def create_quantile_dataloader(cfg, rank=0, world_size=1):
    """Create DataLoader for QuantilePLibDataset."""
    dataset = QuantilePLibDataset(cfg, rank=rank, world_size=world_size)
    loader_cfg = cfg.get("data", {}).get("loader", {})
    batch_size = loader_cfg.get("batch_size", 1)
    num_workers = loader_cfg.get("num_workers", 0)
    pin_memory = loader_cfg.get("pin_memory", False)
    drop_last = loader_cfg.get("drop_last", True)
    shuffle = loader_cfg.get("shuffle", False)
    prefetch_factor = loader_cfg.get("prefetch_factor")

    if dataset._on_gpu:
        if num_workers > 0:
            print("[create_quantile_dataloader] forcing num_workers=0 for GPU data")
        num_workers = 0
        pin_memory = False
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle = False
    # DataLoader rejects prefetch_factor when num_workers == 0 (which _on_gpu forces above).
    extra = {}
    if num_workers > 0 and prefetch_factor is not None:
        extra["prefetch_factor"] = int(prefetch_factor)
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=True if num_workers > 0 else False,
        **extra,
    )
    if sampler is not None:
        dl.set_epoch = lambda e: sampler.set_epoch(e)
    return dl


class MemoryShardLoader:
    """Batches over a QuantilePLibDataset built with data.memory_shard.

    Each batch is one index_select per field on the resident shard -- no worker processes,
    no collate -- so a step costs the gather plus (for a host-RAM shard) one host->GPU copy
    of the batch, which train.py's .to(DEVICE) already does.

    Mirrors the parts of the DataLoader interface train.py and QuantileInfer touch:
    len(), iteration, .dataset, and .sampler.set_epoch (train.py reshuffles through it).
    The permutation is seeded by (seed, epoch, rank), so a resumed chain link replays the
    same order for the same epoch, the way DistributedSampler's seed + epoch does.
    """

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True, seed=0, rank=0, world_size=1):
        self.dataset = dataset
        self.sampler = self
        self._batch_size = int(batch_size)
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._seed = int(seed)
        self._rank = rank
        self._world_size = world_size
        self._epoch = 0

    def set_epoch(self, epoch):
        self._epoch = int(epoch)

    def __len__(self):
        n = len(self.dataset)
        if self._drop_last:
            return n // self._batch_size
        return (n + self._batch_size - 1) // self._batch_size

    def __iter__(self):
        n = len(self.dataset)
        dev = self.dataset._device
        if self._shuffle:
            g = torch.Generator(device=dev)
            g.manual_seed(self._seed + self._epoch * self._world_size + self._rank)
            order = torch.randperm(n, generator=g, device=dev)
        else:
            order = torch.arange(n, device=dev)
        for b in range(len(self)):
            yield self.dataset[order[b * self._batch_size:(b + 1) * self._batch_size]]