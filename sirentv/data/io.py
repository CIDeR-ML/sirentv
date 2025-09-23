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

        # get weighting scheme
        weight_cfg = cfg.get("data", {}).get("dataset", {}).get("weight", {})
        if weight_cfg:
            method = weight_cfg.get("method")
            if method == "vis":
                self.get_weight = self.get_weight_by_vis
                print("[PLibDataLoader] weighting using", method)
                print("[PLibDataLoader] params:", weight_cfg)
            elif method == "bivis":
                self.get_weight = self.get_biweight_by_vis
                print("[PLibDataLoader] weighting using", method)
                print("[PLibDataLoader] params:", weight_cfg)
            else:
                self.get_weight = lambda vis: vis.new_ones(vis.shape, device=device)
                # raise NotImplementedError(f'Weight method {method} is invalid')
            self._weight_cfg = weight_cfg
        else:
            print("[PLibDataLoader] weight = 1")
            self.get_weight = lambda vis: vis.new_ones(vis.shape, device=device)

        # tranform visiblity in pseudo-log scale (default: False)
        xform_params = cfg.get("transform_vis")
        if xform_params:
            print("[PLibDataLoader] using log scale transformaion")
            print("[PLibDataLoader] transformation params", xform_params)

        self.xform_vis, self.inv_xform_vis = partial_xform_vis(xform_params)

        # prepare dataloader
        loader_cfg = cfg.get("data", {}).get("loader")
        geom_cfg = cfg.get("data", {}).get("geometry")
        self._batch_mode = loader_cfg is not None
        self._n_photons = cfg["data"]["dataset"]["weight"].get("n_photon", 200000)

        if self._batch_mode:
            # dataloader in batches
            self._batch_size = loader_cfg.get("batch_size", 1)
            self._shuffle = loader_cfg.get("shuffle", False)
            self._n_pmt = geom_cfg.get("n_pmts", 48)
        # else:
        # returns the whole plib in a single batch
        if not self._is_lazy:
            print("[PLibDataLoader] precomputing full-cache")
            # precompute full-cache only when non-lazy
            n_voxels = len(self._plib)
            vox_ids = torch.arange(n_voxels, device=device)

            meta = self._plib.meta
            pos_raw = meta.voxel_to_coord(vox_ids)
            pos = meta.norm_coord(pos_raw)

            vis = self._plib.vis * self._plib.eff
            vis[:, :self._n_pmt] = vis[:, self._n_pmt:].reshape(vis.shape[0], self._n_pmt, -1).sum(-1)
            w = self.get_weight(vis)
            target = self.xform_vis(vis)

            self._cache = dict(norm_position=pos, raw_position=pos_raw, value=vis, weight=w, target=target)
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

    def get_biweight_by_vis(self, vis):
        factors = self._weight_cfg.get("factor", [1.0, 1.0])
        thresholds = self._weight_cfg.get("threshold", [1e-8, 1e-8])
        idx_slices = self._weight_cfg.get("idx_slices", [[None], [None]])

        w = torch.ones_like(vis)

        min_weight = min(factors) * torch.min(vis[vis > 0])
        for factor, threshold, idx_slice in zip(factors, thresholds, idx_slices):
            w[:, slice(*idx_slice)] = vis[:, slice(*idx_slice)] * factor
        w[w < threshold] = min_weight / 10
        return w

    def __len__(self):
        """
        Number of batches.
        """
        from math import ceil

        if self._batch_mode:
            return ceil(len(self._plib) / self._batch_size)

        return 1

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

            for b in range(len(self)):
                sel = slice(b * self._batch_size, (b + 1) * self._batch_size)

                vox_ids = vox_list[sel]
                if self._is_lazy or self._cache is None:
                    # fetch per-batch on the fly
                    pos_raw = meta.voxel_to_coord(vox_ids)
                    pos = meta.norm_coord(pos_raw)
                    # try fast item access first
                    try:
                        vis = self._plib[vox_ids] / self._n_photons
                    except Exception:
                        vis = self._plib.vis[vox_ids] * self._plib.eff

                    vis = vis.view(vis.shape[0], self._n_pmt, -1)
                    w = self.get_weight(vis)
                    target = self.xform_vis(vis)
                    yield dict(norm_position=pos, raw_position=pos_raw, value=vis, weight=w, target=target)
                else:
                    #vis = self._cache["value"][vox_ids]
                    # print(self._cache["target"][vox_ids][0,48:])
                    output = dict(
                        norm_position=self._cache["norm_position"][vox_ids],
                        raw_position=self._cache["raw_position"][vox_ids],
                        value=self._cache["value"][vox_ids],
                        weight=self._cache["weight"][vox_ids],
                        target=self._cache["target"][vox_ids],
                    )
                    yield output
        else:
            if self._is_lazy or self._cache is None:
                # build on demand without precomputing entire cache in ctor
                n_voxels = len(self._plib)
                vox_ids = torch.arange(n_voxels, device=self.device)
                meta = self._plib.meta
                pos = meta.norm_coord(meta.voxel_to_coord(vox_ids))
                try:
                    vis = self._plib[vox_ids]
                except Exception:
                    vis = self._plib.vis

                w = self.get_weight(vis)
                target = self.xform_vis(vis)
                yield dict(
                    norm_position=pos.to(self.device),
                    raw_position=pos_raw.to(self.device),
                    value=vis.to(self.device),
                    weight=w.to(self.device),
                    target=target.to(self.device),
                )
            else:
                yield self._cache