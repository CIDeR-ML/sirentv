import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from slar.transform import partial_xform_vis
from .plib import TVPhotonLib

class PLibDataLoader:
    '''
    A fast implementation of PhotonLib dataloader.
    '''
    def __init__(self, cfg, device=None):
        '''
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
        '''

        # load plib to device
        self._plib = TVPhotonLib.load(cfg).to(device)
        
        # get weighting scheme
        weight_cfg = cfg.get('data',{}).get('dataset',{}).get('weight', {})
        if weight_cfg:
            method = weight_cfg.get('method')
            if method == 'vis':
                self.get_weight = self.get_weight_by_vis
                print('[PLibDataLoader] weighting using', method)
                print('[PLibDataLoader] params:', weight_cfg)
            elif method == 'bivis':
                self.get_weight = self.get_biweight_by_vis
                print('[PLibDataLoader] weighting using', method)
                print('[PLibDataLoader] params:', weight_cfg)
            else:
                self.get_weight = lambda vis : torch.tensor(1., device=device)
                # raise NotImplementedError(f'Weight method {method} is invalid')
            self._weight_cfg = weight_cfg
        else:
            print('[PLibDataLoader] weight = 1')
            self.get_weight = lambda vis : torch.tensor(1., device=device)

        # tranform visiblity in pseudo-log scale (default: False)
        xform_params = cfg.get('transform_vis')
        if xform_params:
            print('[PLibDataLoader] using log scale transformaion')
            print('[PLibDataLoader] transformation params',xform_params)

        self.xform_vis, self.inv_xform_vis = partial_xform_vis(xform_params)

        # prepare dataloader
        loader_cfg = cfg.get('data',{}).get('loader')
        self._batch_mode = loader_cfg is not None

        if self._batch_mode:
            # dataloader in batches
            self._batch_size = loader_cfg.get('batch_size', 1)
            self._shuffle = loader_cfg.get('shuffle', False)
        # else:
        # returns the whole plib in a single batch
        n_voxels = len(self._plib)
        vox_ids = torch.arange(n_voxels, device=device)

        meta = self._plib.meta
        pos = meta.norm_coord(meta.voxel_to_coord(vox_ids))

        vis = self._plib.vis
        w = self.get_weight(vis)
        target = self.xform_vis(vis)

        self._cache = dict(position=pos, value=vis, weight=w, target=target)

    @property
    def device(self):
        return self._plib.device
    
    def get_weight_by_vis(self, vis):
        '''
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
        '''
        factor = self._weight_cfg.get('factor', 1.)
        threshold = self._weight_cfg.get('threshold', 1e-8)
        w = vis * factor
        w[w<threshold] = 1.
        return w

    def get_biweight_by_vis(self, vis):
        factors = self._weight_cfg.get('factor', [1., 1.])
        thresholds = self._weight_cfg.get('threshold', [1e-8, 1e-8])
        idx_slices = self._weight_cfg.get('idx_slices', [[None], [None]])
        
        w = torch.ones_like(vis)

        min_weight = min(factors) * torch.min(vis[vis > 0])
        for (factor, threshold, idx_slice) in zip(factors, thresholds, idx_slices):
            w[:, slice(*idx_slice)] = vis[:, slice(*idx_slice)] * factor

        w[w < threshold] = min_weight / 10
        return w

    def __len__(self):
        '''
        Number of batches.
        '''
        from math import ceil
        if self._batch_mode:
            return ceil(len(self._plib) / self._batch_size)

        return 1
        
    def __iter__(self):
        '''
        Generator of batch data.

        For non-batch mode, the whole photon lib is returned in a single entry
        from the cache.
        '''
        if self._batch_mode:
            meta = self._plib.meta
            n_voxels = len(self._plib)
            if self._shuffle:
                vox_list = torch.randperm(n_voxels, device=self.device)
            else:
                vox_list = torch.arange(n_voxels, device=self.device)

            for b in range(len(self)):
                sel = slice(b*self._batch_size, (b+1)*self._batch_size)

                vox_ids = vox_list[sel]
                # pos = meta.norm_coord(meta.voxel_to_coord(vox_ids))
                # vis = self._plib[vox_ids]
                # w = self.get_weight(vis)
                # target = self.xform_vis(vis)
                output = dict(
                    position=self._cache["position"][vox_ids],
                    value=self._cache["value"][vox_ids],
                    weight=self._cache["weight"][vox_ids],
                    target=self._cache["target"][vox_ids],
                )

                # output = dict(position=pos, value=vis, weight=w, target=target)
                yield output
        else:
            yield self._cache
