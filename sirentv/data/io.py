import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from slar.transform import partial_xform_vis
from photonlib import PhotonLib
from photonlib.meta import VoxelMeta
import h5py
import numpy as np

class PLibDataLoader:
    """
    A fast implementation of PhotonLib dataloader.
    """

    def __init__(self, cfg, device=None):
        """
        Constructor.

        Arguments
        ---------
        cfg: dict
            Config dictionary. See "Examples" bewlow.

        device: torch.device (optional)
            Device for the returned data. Default: None.

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
        # determine lazy mode from PhotonLib or config
        self._is_lazy = cfg.get("photonlib", {}).get("lazy", False)
        #bool(
        #getattr(self._plib, "lazy", False)
        #or cfg.get("photonlib", {}).get("lazy", False)
        #))
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

        if self._batch_mode and self._drop_last and self._batch_size > 0:
            remainder = self._effective_voxels % self._batch_size
            if remainder:
                self._effective_voxels -= remainder

        if self._effective_voxels < 0:
            self._effective_voxels = 0
        

        # returns the whole plib in a single batch
        if not self._is_lazy:
            print("[PLibDataLoader] precomputing full-cache")
            # precompute full-cache only when non-lazy
            n_voxels = self._effective_voxels
            vox_ids = torch.arange(n_voxels, device=device)

            meta = self._plib.meta
            pos = meta.voxel_to_coord(vox_ids)

            vis = self._plib.vis * self._plib.eff
            vis[:, :self._n_pmt] = vis[:, self._n_pmt:].reshape(vis.shape[0], self._n_pmt, -1).sum(-1)
            # w = self.get_weight(vis)
            target = self.xform_vis(vis)

            self._cache = dict(position=pos, value=vis, target=target)
        else:
            # in lazy mode, do not create full-cache
            self._cache = None

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
                return self._effective_voxels // self._batch_size
            else:
                return ceil(self._effective_voxels / self._batch_size) if self._effective_voxels else 0

        return 1 if self._effective_voxels else 0

    def __iter__(self):
        """
        Generator of batch data.

        For non-batch mode, the whole photon lib is returned in a single entry
        from the cache.
        """
        if self._batch_mode:
            meta = self._plib.meta
            n_voxels = len(self._plib)
            if self._shuffle:
                vox_list = torch.randperm(n_voxels, device=self.device)
            else:
                vox_list = torch.arange(n_voxels, device=self.device)

            vox_list = vox_list[: self._effective_voxels]

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
            if self._is_lazy or self._cache is None:
                # build on demand without precomputing entire cache in ctor
                n_voxels = self._effective_voxels
                if n_voxels == 0:
                    return
                vox_ids = torch.arange(n_voxels, device=self.device)
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