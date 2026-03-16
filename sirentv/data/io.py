import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from slar.transform import partial_xform_vis
from photonlib import PhotonLib
from photonlib.meta import VoxelMeta
import h5py
import numpy as np
import os

from sirentv.utils.misc import compute_gradients_from_positions
from sirentv.data.builder import DATASETS
from sirentv.utils.transform import pdf_to_cdf

@DATASETS.register_module()
class PLibDataset(Dataset):
    """
    Map-style dataset for PhotonLib.
    Requires efficient random access to photon library.
    """

    def __init__(self, cfg, rank=0, world_size=1):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size

        # Load photon library
        self._is_lazy = cfg.get("photonlib", {}).get("lazy", False)
        self._plib = PhotonLib.load(cfg, self._is_lazy)

        # Transform
        xform_params = cfg.get("transform_vis")
        self.xform_vis, self.inv_xform_vis = partial_xform_vis(xform_params)

        if xform_params and rank == 0:
            print("[PLibDataset] using log scale transformation")
            print("[PLibDataset] transformation params", xform_params)

        # Data config
        self.data_cfg = cfg["data"]
        self._n_photons = self.data_cfg.get("n_photon", 200000)
        self._n_pmt = self.data_cfg.get("n_pmt", 81)

        total_voxels = len(self._plib)
        max_len = self.data_cfg.get("max_len", -1)
        if max_len is None or max_len < 0:
            max_len = total_voxels
        effective_voxels = min(total_voxels, max_len)

        self.indices = torch.arange(0, effective_voxels, dtype=torch.long)

        if rank == 0:
            print(f"[PLibDataset] Total voxels: {effective_voxels}")

        self._model_mode = cfg.get("model", {}).get("mode", "pdf").lower()

        self.use_gradient = self.data_cfg.get("use_gradient_supervision", False)

        if self.use_gradient:
            self._precompute_gradients(rank, world_size)

    def _precompute_gradients(self, rank, world_size):
        """
        Precompute gradient magnitudes for all voxels and PMTs
        Only rank 0 computes, then broadcasts to all ranks
        """

        grad_cache_file = self.data_cfg.get("gradient_cache_file", None)

        if grad_cache_file and os.path.exists(grad_cache_file):
            if rank == 0:
                print(f"[PLibDataset] Loading precomputed gradients from {grad_cache_file}")
            cache = torch.load(grad_cache_file)
            self.grad_mags_linear = cache['grad_mags_linear']
            self.grad_mags_transformed = cache['grad_mags_transformed']
            if rank == 0:
                print(f"[PLibDataset] Loaded gradient magnitudes: {self.grad_mags_transformed.shape}")
            return

        if rank == 0:
            #print("[PLibDataset] Precomputing gradient magnitudes...")
            #print("[PLibDataset] This will take a while but only happens once...")

            # Get all positions and visibilities
            n_voxels = len(self.indices)
            positions = np.zeros((n_voxels, 3), dtype=np.float32)
            visibility_linear = np.zeros((n_voxels, self._n_pmt), dtype=np.float32)

            # Load data
            meta = self._plib.meta
            for i, idx in enumerate(tqdm(self.indices, desc="Loading data")):
                vox_ids = torch.tensor([idx])
                pos_raw = meta.voxel_to_coord(vox_ids)
                positions[i] = pos_raw.cpu().numpy()

                try:
                    vis = self._plib[vox_ids] / self._n_photons
                except Exception:
                    vis = (self._plib.vis[vox_ids] * self._plib.eff)

                vis_summed = vis.sum(dim=-1) #(n_pmt, )
                visibility_linear[i] = vis_summed.cpu().numpy()

            # Compute gradients for each PMT
            grad_mags_linear = np.zeros((n_voxels, self._n_pmt), dtype=np.float32)

            for pmt_idx in tqdm(range(self._n_pmt), desc="Computing gradients (linear)"):
                _, _, _, grad_mag_linear, _, _, _ = compute_gradients_from_positions(
                    positions,
                    visibility_linear,
                    pmt_idx
                )
                grad_mags_linear[:, pmt_idx] = grad_mag_linear

            self.grad_mags_linear = torch.tensor(grad_mags_linear, dtype=torch.float32)

            # Also compute gradients in TRANSFORMED space
            visibility_transformed = np.zeros((n_voxels, self._n_pmt), dtype=np.float32)

            for i in range(n_voxels):
                vis_torch = torch.tensor(visibility_linear[i], dtype=torch.float32)
                vis_xform = self.xform_vis(vis_torch)
                visibility_transformed[i] = vis_xform.cpu().numpy()

            grad_mags_transformed = np.zeros((n_voxels, self._n_pmt), dtype=np.float32)

            for pmt_idx in tqdm(range(self._n_pmt), desc="Computing gradients (transformed)"):
                _, _, _, grad_mag, _, _, _ = compute_gradients_from_positions(
                    positions,
                    visibility_transformed,
                    pmt_idx
                )
                grad_mags_transformed[:, pmt_idx] = grad_mag

            self.grad_mags_transformed = torch.tensor(grad_mags_transformed, dtype=torch.float32)

            # Save to cache if specified
            if grad_cache_file:
                print(f"[PLibDataset] Saving gradients to {grad_cache_file}")
                os.makedirs(os.path.dirname(grad_cache_file), exist_ok=True)
                torch.save({
                    'grad_mags_linear': self.grad_mags_linear,
                    'grad_mags_transformed': self.grad_mags_transformed
                }, grad_cache_file)

            print(f"[PLibDataset] Gradient precomputation complete")
            print(f"  Linear: {self.grad_mags_linear.shape}")
            print(f"  Transformed: {self.grad_mags_transformed.shape}")

        else:
            # Non-rank-0 processes wait
            self.grad_mags_linear = None
            self.grad_mags_transformed = None

        # Broadcast from rank 0 to all other ranks
        if world_size > 1:
            import torch.distributed as dist
            if rank == 0:
                shape = torch.tensor(self.grad_mags_linear.shape, dtype=torch.long)
            else:
                shape = torch.zeros(2, dtype=torch.long)

            dist.broadcast(shape, src=0)

            if rank != 0:
                self.grad_mags_linear = torch.zeros(tuple(shape.tolist()), dtype=torch.float32)
                self.grad_mags_transformed = torch.zeros(tuple(shape.tolist()), dtype=torch.float32)

            dist.broadcast(self.grad_mags_linear, src=0)
            dist.broadcast(self.grad_mags_transformed, src=0)

            if rank == 0:
                print("[PLibDataset] Gradients broadcasted to all ranks")

    def __len__(self):
        return len(self.indices)

    def get_weight_by_vis(self, vis):
        """
        Weight by inverse visibility, `weight  = 1/vis * factor`.
        Weights below `threshold` are set to 1.

        Arguments
        ---------
        vis: torch.Tensor
            Visibility values.

        Returns
        -------
        w: torch.Tensor
            Weight values with `w.shape == vis.shape`.
        """
        factor = self._weight_cfg.get("factor", 1.0)
        threshold = self._weight_cfg.get("threshold", 1e-8)
        w = vis * factor
        w[w < threshold] = 1.0
        return w

    def __getitem__(self, idx):
        """
        Get a single voxel's data in standardized format.

        Returns:
            dict with "position", "target" (loss keys), "meta" (plotting data)
        """
        vox_ids = self.indices[idx].unsqueeze(0)

        meta = self._plib.meta
        pos_raw = meta.voxel_to_coord(vox_ids)
        if pos_raw.dim() == 1:
            pos_raw = pos_raw.unsqueeze(0)

        try:
            vis = self._plib[vox_ids] / self._n_photons
        except Exception:
            vis = (self._plib.vis[vox_ids] * self._plib.eff)

        if hasattr(self._plib, 'vis_mask') and self._plib.vis_mask is not None:
            modified_mask = self._plib.vis_mask[vox_ids]
        else:
            modified_mask = torch.zeros(1, dtype=torch.bool)
        vis = vis.view(self._n_pmt, -1)  # (n_pmt, n_time)
        vis_transformed = self.xform_vis(vis)

        # Compute visibility by summing over time
        v_linear = vis.sum(-1)  # (n_pmt,)
        v = self.xform_vis(v_linear)  # transformed

        # Timing target: PDF or CDF depending on model mode
        if self._model_mode == "cdf":
            t = pdf_to_cdf(vis)  # CDF in linear domain
            t_linear = t
        else:
            t = vis_transformed  # PDF in transformed domain
            t_linear = vis  # PDF in linear domain

        target = {"t": t, "v": v}
        meta_dict = {
            "t_linear": t_linear,
            "v_linear": v_linear,
            "v_mask": ~modified_mask,
        }

        if self.use_gradient:
            actual_idx = self.indices[idx].item()
            target["grad_mags_transformed"] = self.grad_mags_transformed[actual_idx]

        return {
            "position": pos_raw.squeeze(0),
            "target": target,
            "meta": meta_dict,
        }

def create_dataloader(cfg, rank=0, world_size=1):
    """Create DataLoader with map-style dataset."""
    # Create dataset
    dataset = PLibDataset(cfg, rank=rank, world_size=world_size)
    # Get loader config
    loader_cfg = cfg.get("data", {}).get("loader", {})
    batch_size = loader_cfg.get("batch_size", 1)
    num_workers = loader_cfg.get("num_workers", 0)
    pin_memory = loader_cfg.get("pin_memory", False)
    drop_last = loader_cfg.get("drop_last", True)
    shuffle = loader_cfg.get("shuffle", False)

    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
            seed = 0
        )
        shuffle = False
    else:
        sampler = None
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last if sampler is None else False,
        shuffle=shuffle,
        persistent_workers=True if num_workers > 0 else False,
    )

    dataloader.xform_vis = dataset.xform_vis
    dataloader.inv_xform_vis = dataset.inv_xform_vis

    return dataloader

class PLibDataLoader:
    """
    A fast implementation of PhotonLib dataloader.
    """

    def __init__(self, cfg, device=None, rank=0, world_size=1):
        """
        Constructor.

        Arguments
        ---------
        cfg: dict
            Config dictionary. See "Examples" bewlow.
        device: torch.device (optional)
            Device for the returned data. Default: None.
        rank: int
            Process rank for distributed training
        world_size: int
            Total number of processes

        Examples
        --------
        This is an example configuration in yaml format.

        ```
                photonlib:
                        filepath: plib_file.h5

                data:
                        dataset:
                                weight:
                                        method: vis
                                        factor: 1000000.0
                                        threshold: 1.0e-08
                        loader:
                                batch_size: 500
                                shuffle: true

        transform_vis:
            eps: 1.0e-05
            sin_out: false
            vmax: 1.0
                ```

        The `photonlib` section provide the input file of `PhotonLib`.

        [Optional] The `weight` subsection is the weighting scheme. Supported
        schemes are:

        1. `vis`, where `weight ~ 1/vis * factor`.  Weights below `threshold`
        are set to one.
        2. To-be-implemented.

        [Optional] The `loader` subsection mimics pytorch's `DataLoader` class,
        however, only `batch_size` and `shuffle` options are implemented.  If
        `loader` subsection is absent, the data loader returns the whole photon
        lib in a single entry.

        [Optional] The `transform_vis` subsection uses `log(vis+eps)` in the
        training. The final output is scaled to `[0,1]`.
        """

        self.rank = rank
        self.world_size = world_size
        self._current_epoch = 0

        # determine lazy mode from PhotonLib or config
        self._is_lazy = cfg.get("photonlib", {}).get("lazy", False)
        # load plib to device
        self._plib = PhotonLib.load(cfg, self._is_lazy).to(device)

        # tranform visiblity in pseudo-log scale (default: False)
        xform_params = cfg.get("transform_vis")
        if xform_params:
            print("[PLibDataLoader] using log scale transformaion")
            print("[PLibDataLoader] transformation params", xform_params)

        self.xform_vis, self.inv_xform_vis = partial_xform_vis(xform_params)

        # prepare dataloader
        data_cfg = cfg["data"]
        loader_cfg = data_cfg.get("loader")
        geom_cfg = data_cfg.get("geometry")
        self._batch_mode = loader_cfg is not None
        self._n_photons = data_cfg.get("n_photon", 200000)
        self._n_pmt = data_cfg.get("n_pmt", 81)
        self._drop_last = False
        if self._batch_mode:
            # dataloader in batches
            self._batch_size = loader_cfg.get("batch_size", 1)
            self._shuffle = loader_cfg.get("shuffle", False)
            self._drop_last = loader_cfg.get("drop_last", True)

        max_len = data_cfg.get("max_len", -1)
        if max_len is None:
            max_len = -1
        try:
            self._max_len = int(max_len)
        except (TypeError, ValueError):
            raise ValueError("max_len must be an integer, -1, or None") from None
        if self._max_len < 0:
            self._max_len = -1

        self._total_voxels = len(self._plib)
        if self._max_len == -1:
            self._effective_voxels = self._total_voxels
        else:
            self._effective_voxels = min(self._total_voxels, self._max_len)

        if self.world_size > 1:
            self._voxels_per_rank = self._effective_voxels // self.world_size
            remainder = self._effective_voxels % self.world_size
            # Distribute remainder to first few ranks
            if self.rank < remainder:
                self._voxels_per_rank += 1
                self._start_idx = self.rank * self._voxels_per_rank
            else:
                self._start_idx = self.rank * self._voxels_per_rank + remainder

            self._end_idx = self._start_idx + self._voxels_per_rank

            if rank == 0:
                print(f"[PLibDataLoader] Distributed mode: {world_size} processes")
                print(f"[PLibDataLoader] Total voxels: {self._effective_voxels}")
                print(f"[PLibDataLoader] Voxels per rank: ~{self._effective_voxels // world_size}")
        else:
            self._voxels_per_rank = self._effective_voxels
            self._start_idx = 0
            self._end_idx = self._effective_voxels

        if self._batch_mode and self._drop_last and self._batch_size > 0:
            remainder = self._voxels_per_rank % self._batch_size
            if remainder:
                self._voxels_per_rank -= remainder
                self._end_idx = self._start_idx + self._voxels_per_rank

        if self._voxels_per_rank < 0:
            self._voxels_per_rank = 0

        self._create_indices()
        # returns the whole plib in a single batch
        if not self._is_lazy and not self._batch_mode:
            if rank == 0:
                print("[PLibDataLoader] precomputing full-cache")
            # precompute full-cache only when non-lazy
            self.cache = self._build_cache()
        else:
            # in lazy mode, do not create full-cache
            self._cache = None

    def _create_indices(self):
        """Create the list of voxel indices for this rank."""
        # Full index range for this rank
        self._full_indices = torch.arange(
            self._start_idx,
            self._end_idx,
            dtype=torch.long
        )

    def set_epoch(self, epoch):
        """
        Set the epoch for the dataloader to ensure proper shuffling in distributed mode.

        Arguments
        ---------
        epoch: int
            Current epoch number
        """
        self._current_epoch = epoch

    def _get_shuffled_indices(self):
        """
        Get shuffled indices based on current epoch.
        Uses the same random seed across all ranks for consistency.
        """
        if self._shuffle:
            # Create generator with epoch-based seed for reproducibility
            generator = torch.Generator()
            generator.manual_seed(self._current_epoch)

            # Shuffle indices for this rank
            perm = torch.randperm(len(self._full_indices), generator=generator)
            return self._full_indices[perm]
        else:
            return self._full_indices

    def _build_cache(self):
        """Build cache for the voxels assigned to this rank."""
        if self._voxels_per_rank == 0:
            return None

        vox_ids = self._full_indices
        meta = self._plib.meta
        pos = meta.voxel_to_coord(vox_ids)

        vis = self._plib.vis * self._plib.eff
        # Select only the voxels for this rank
        vis = vis[vox_ids]
        vis = vis.reshape(vis.shape[0], self._n_pmt, -1)

        target = self.xform_vis(vis)

        return dict(position=pos, value=vis, target=target)

    @property
    def device(self):
        return self._plib.device

    def get_weight_by_vis(self, vis):
        """
        Weight by inverse visibility, `weight  = 1/vis * factor`.
        Weights below `threshold` are set to 1.

        Arguments
        ---------
        vis: torch.Tensor
            Visibility values.

        Returns
        -------
        w: torch.Tensor
            Weight values with `w.shape == vis.shape`.
        """
        factor = self._weight_cfg.get("factor", 1.0)
        threshold = self._weight_cfg.get("threshold", 1e-8)
        w = vis * factor
        w[w < threshold] = 1.0
        return w

    def __len__(self):
        """
        Number of batches.
        """
        from math import ceil

        if self._batch_mode:
            if self._drop_last:
                return self._voxels_per_rank // self._batch_size
            else:
                return ceil(self._voxels_per_rank / self._batch_size) if self._voxels_per_rank else 0

        return 1 if self._voxels_per_rank else 0

    def __iter__(self):
        """
        Generator of batch data.

        For non-batch mode, the whole photon lib is returned in a single entry
        from the cache.
        """
        if self._batch_mode:
            meta = self._plib.meta
            vox_list = self._get_shuffled_indices()
            for b in range(len(self)):
                sel = slice(b * self._batch_size, (b + 1) * self._batch_size)

                vox_ids = vox_list[sel]
                if vox_ids.numel() == 0:
                    break
                if self._is_lazy or self._cache is None:
                    # fetch per-batch on the fly
                    pos_raw = meta.voxel_to_coord(vox_ids)
                    # pos = meta.norm_coord(pos_raw)
                    # try fast item access first
                    try:
                        vis = self._plib[vox_ids] / self._n_photons
                    except Exception:
                        vis = self._plib.vis[vox_ids] * self._plib.eff

                    vis = vis.view(vis.shape[0], self._n_pmt, -1)
                    #w = self.get_weight(vis)
                    target = self.xform_vis(vis)

                    if pos_raw.dim() == 1:
                        pos_raw = pos_raw.unsqueeze(0)
                    yield dict(position=pos_raw, target_linear=vis, target=target)
                else:
                    #vis = self._cache["value"][vox_ids]
                    # print(self._cache["target"][vox_ids][0,48:])
                    output = dict(
                        position=self._cache["position"][vox_ids],
                        target_linear=self._cache["value"][vox_ids],
                        target=self._cache["target"][vox_ids],
                    )
                    yield output
        else:
            # Non-batch mode: return all data for this rank
            if self._is_lazy or self._cache is None:
                # build on demand without precomputing entire cache in ctor
                if self._voxels_per_rank == 0:
                    return
                vox_ids = self._full_indices
                meta = self._plib.meta
                pos = meta.voxel_to_coord(vox_ids)
                try:
                    vis = self._plib[vox_ids]
                except Exception:
                    vis = self._plib.vis

                #w = self.get_weight(vis)
                target = self.xform_vis(vis)

                if pos.dim() == 1:
                    pos = pos.unsqueeze(0)
                yield dict(
                    position=pos.to(self.device),  
                    target_linear=vis.to(self.device),
                    #weight=w.to(self.device),
                    target=target.to(self.device),
                )
            else:
                yield self._cache