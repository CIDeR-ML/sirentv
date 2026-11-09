"""Reader and dataset for the quantile photon library H5 (vis, analytical_t0, quantiles)."""

import os
import time

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from slar.transform import partial_xform_vis
from photonlib.meta import VoxelMeta

from sirentv.data.builder import DATASETS
from sirentv.data.common import grad_target_options, precompute_grad_targets, select_voxel_indices


class QuantilePLib:
    """Reader for the quantile waveform photon library H5 (vis, t0, quantiles); one detector half."""

    FIELDS = ("vis", "t0", "quantiles")

    def __init__(self, filepath, lazy=True, mode="quantile", combine_every_quantile=1, device=None, log_quantile_C=None):
        self._path = filepath
        self._lazy = lazy
        self._device = device or torch.device("cpu")
        self._file = None
        self._mode = mode

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
            self.meta = VoxelMeta(numvox, torch.column_stack((min_xyz, max_xyz)))

        self.clear_prefetch()

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
    def load(cls, filepath, lazy=True, mode="quantile", combine_every_quantile=1, device=None, log_quantile_C=None):
        return cls(filepath, lazy=lazy, mode=mode, combine_every_quantile=combine_every_quantile, device=device, log_quantile_C=log_quantile_C)

    def read_strided(self, start, stop, step, fields=None):
        """Rows start:stop:step of the requested fields, as ONE hyperslab per field (lazy reader only).

        For loading a rank shard up front: _read_slice's fancy index is a point selection,
        far slower for tens of thousands of rows.
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
                # h5py needs sorted, unique indices
                uniq, inv = np.unique(voxel_ids, return_inverse=True)
                arr = ds[uniq][inv]

        if key == "quantiles":
            arr = arr[..., ::self._combine_every_quantile]
        return torch.from_numpy(np.asarray(arr)).to(self._device)

    def __getitem__(self, voxel_ids, fields=None):
        """The requested fields (default: all of FIELDS) for these voxel(s).

        Reading only what a run needs matters: one voxel of quantiles is 162 KiB against
        ~0.3 KiB for vis or t0 (see QuantilePLibDataset._fields_for_targets).
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
        """pos: (N, 3) -> {vis, t0, quantiles}, from the prefetch cache when one is set."""
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

    @property
    def mode(self):
        """'quantile' or 'log_quantile'."""
        return self._mode

    def reconstruct_aligned_cdf(self, u_grid, quantiles, t0=None):
        """quantiles (..., K) at levels u_grid -> CDF (..., n_bins) on [0, t_max_ns].
        t0 (..., n_pmt) [ns] is added back for t0-aligned quantiles (photonlib.align_quantiles_to_t0)."""
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
        """quantiles (..., K), possibly log_quantile -> linear-ns quantile times (..., K), clamped at 0.
        t0 (..., n_pmt) [ns] is added back for t0-aligned quantiles."""
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
    """Map-style dataset over QuantilePLib: position -> target (v, t0, quantiles, quantiles_mask[, *_grad_frob]).

    v = xform_vis(vis / n_photon), t0 = log(t0), quantiles via _xform_quantiles; quantiles_mask
    (vis_raw > 0) keeps the LUT's NaN quantiles of invisible pairs out of the loss.
    meta carries v_linear (raw counts) and t0_raw. data.target_keys limits the targets (and
    the LUT fields read); data.memory_shard keeps this rank's shard resident (MemoryShardLoader).
    """

    def __init__(self, cfg, rank=0, world_size=1):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        plib_cfg = cfg.get("quantile_plib", cfg.get("photonlib", {}))
        filepath = plib_cfg.get("filepath")
        lazy = plib_cfg.get("lazy", True)
        combine_every_quantile = plib_cfg.get("combine_every_quantile", 1)
        self._vis_eps = plib_cfg.get("vis_eps", 1e-6)
        self._n_photon = float(plib_cfg.get("n_photon", 1.0))
        mode = plib_cfg.get("mode", "quantile")
        self._device = torch.device(plib_cfg.get("device", "cpu"))
        self._on_gpu = self._device.type == "cuda"

        self._quantile_plib = QuantilePLib.load(
            filepath, lazy=(lazy and not self._on_gpu), mode=mode,
            combine_every_quantile=combine_every_quantile,
            log_quantile_C=plib_cfg.get("log_quantile_C"),
        )
        self._log_quantile = mode == "log_quantile"
        if self._log_quantile:
            self._log_quantile_C = self._quantile_plib._log_quantile_C
        self._align_t0 = bool(plib_cfg.get("align_quantiles_to_t0", False))
        self._xform_vis, self._inv_xform_vis = partial_xform_vis(cfg.get("transform_vis", {}))

        data_cfg = cfg.get("data", {})
        self.indices = select_voxel_indices(
            data_cfg, len(self._quantile_plib), self._quantile_plib.pos, rank, "QuantilePLibDataset")

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

        train_cfg = cfg.get("train", {})
        self._grad_keys = [k for k in train_cfg.get("grad_supervision_keys", []) if k in ("v", "quantiles")]
        self._grad_opts = grad_target_options(train_cfg)
        if self._grad_keys:
            qlib = self._quantile_plib
            self._grad_targets = precompute_grad_targets(
                qlib, self.indices.numpy(), self._grad_keys, "quantiles",
                self._read_v, self._read_quantiles, qlib._n_quantiles // qlib._combine_every_quantile,
                train_cfg, self._grad_opts, self._grad_cache_meta(), self._n_photon,
                "QuantilePLibDataset",
            )

        # data.memory_shard: this rank holds indices[rank::world_size] resident, so every rank
        # sees the whole detector and each epoch visits every voxel once, but a voxel stays on
        # one rank for the whole run (DistributedSampler re-deals every epoch).
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
        elif self._on_gpu or not lazy:
            self._precompute_targets()

    def _grad_cache_meta(self):
        """Every config choice the cached gradient targets depend on (a mismatch recomputes them)."""
        from sirentv.utils.misc import GRAD_CACHE_FORMAT_VERSION

        xform_cfg = self.cfg.get("transform_vis", {})
        plib_cfg = self.cfg.get("quantile_plib", self.cfg.get("photonlib", {}))
        # only when set, so caches from before these options existed still match
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
            "n_projections": self._grad_opts["n_projections"],
            "seed": 0,
        }

    def _read_v(self, rows):
        vis_norm = self._quantile_plib[rows]["vis"].float() / self._n_photon
        return self._xform_vis(vis_norm).numpy()[:, :, None]  # (chunk, n_pmt, 1)

    def _read_quantiles(self, rows):
        d = self._quantile_plib.__getitem__(rows, fields=("quantiles", "t0"))
        quantiles = self._xform_quantiles(d["quantiles"].float(), d["t0"].float())
        # invalid pairs are already masked out of the finite difference; this keeps NaN out of logs
        return torch.nan_to_num(quantiles, nan=0.0).numpy()  # (chunk, n_pmt, K)

    def _precompute_targets(self):
        """Store every transformed target on self._device."""
        idx = self.indices.numpy()
        self._pos = torch.from_numpy(self._quantile_plib.pos[idx]).float().to(self._device)

        data = self._quantile_plib[idx]
        vis_raw = data["vis"].float()
        t0_raw = data["t0"].float()
        quantiles = data["quantiles"].float()

        self._v = self._xform_vis(vis_raw / self._n_photon).to(self._device)
        self._t0 = torch.log(t0_raw.clamp(min=1e-3)).to(self._device)
        self._quantiles = self._xform_quantiles(quantiles, t0_raw).to(self._device)
        self._vis_raw = vis_raw.to(self._device)
        self._t0_raw = t0_raw.to(self._device)

        self._quantile_plib.close()
        print(f"[QuantilePLibDataset] precomputed {len(self.indices)} voxels on {self._device}")

    def _xform_quantiles(self, quantiles, t0_raw):
        """Raw LUT quantiles (..., n_pmt, K) in ns -> the network's target domain.

        The one place this transform lives (eager, lazy, shard and grad-target paths).
        With align_quantiles_to_t0 the onset is removed first, so the log acts on the delay.
        """
        if self._align_t0:
            quantiles = quantiles - t0_raw.unsqueeze(-1)
        if self._log_quantile:
            quantiles = torch.log10(quantiles.clamp(min=0.0) + self._log_quantile_C)
        return quantiles

    def _precompute_shard(self, chunk_size):
        """Load this rank's strided shard into self._device, chunk by chunk via read_strided.

        Transformed chunks go straight into preallocated storage, so the transient footprint is
        one chunk (_precompute_targets' full-size copies exceed node RAM at 4 ranks per node).
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
                # rate of first quantiles below analytical_t0 (clamped to 0 by _xform_quantiles)
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
        """LUT fields needed for the requested targets. quantiles needs vis too (quantiles_mask
        comes from the raw visibility) and, aligned, t0 (the target is Q - analytical_t0)."""
        if target_keys is None:
            return set(QuantilePLib.FIELDS)
        need = {"v": {"vis"}, "t0": {"t0"}, "quantiles": {"quantiles", "vis"}}
        if align_t0:
            need["quantiles"] = {"quantiles", "vis", "t0"}
        fields = set()
        for k in target_keys:
            fields |= need[k]
        return fields

    def _lazy_targets(self, data):
        """LUT fields read for one voxel or a batch -> (target, meta) dicts."""
        want = self._target_keys
        target, meta = {}, {}
        vis_raw = data["vis"].float() if "vis" in data else None  # (..., n_pmt)
        if vis_raw is not None:
            meta["v_linear"] = vis_raw
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
        return target, meta

    def __getitem__(self, idx):
        if hasattr(self, "_v"):
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
                "meta": {"v_linear": self._vis_raw[idx], "t0_raw": self._t0_raw[idx]},
            }

        vox_id = self.indices[idx].item()
        pos = torch.from_numpy(self._quantile_plib.pos[vox_id]).float()
        target, meta = self._lazy_targets(self._quantile_plib.__getitem__(vox_id, fields=self._read_fields))
        for key in self._grad_keys:
            target[f"{key}_grad_frob"] = self._grad_targets[key][idx]
        return {"position": pos, "target": target, "meta": meta}

    def __getitems__(self, indices):
        """A whole batch with ONE read per field (see CompressedPLibDataset.__getitems__); for
        162 KiB/voxel quantiles that is ~1.2 GB/s against ~79 MB/s per sample."""
        if hasattr(self, "_v"):
            return [self[i] for i in indices]

        idx = np.asarray(indices)
        vox = self.indices[torch.as_tensor(idx, dtype=torch.long)].numpy()
        cols, metacols = self._lazy_targets(self._quantile_plib.__getitem__(vox, fields=self._read_fields))
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


class MemoryShardLoader:
    """Batches over a QuantilePLibDataset built with data.memory_shard: one index_select per
    field on the resident shard, no workers, no collate.

    Provides what train.py and QuantileInfer use from a DataLoader: len(), iteration,
    .dataset and .sampler.set_epoch. The order is seeded by (seed, epoch, rank), so a resumed
    run replays the same order for the same epoch.
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
