import torch
import numpy as np
from photonlib.meta import VoxelMeta
import h5py

class TVPhotonLib:
    def __init__(self, meta: VoxelMeta, vis: torch.Tensor, eff: float = 1.0,
                 vis_slice =slice(None, 48),
                 time_slice=slice(48, None)):
        """
        Constructor

        Parameters
        ----------
        meta : VoxelMeta
            Defines the volume and its voxelization scheme for a photon library
        vis  : torch.Tensor
            Visibility map as 3D array with shape (n_voxels, n_pmts, n_values)
        eff  : float or torch.Tensor
            Overall scaling factor for the visibility. If tensor, should be broadcastable to vis shape.
        """
        self._meta = meta
        self._eff = torch.as_tensor(eff, dtype=torch.float32)
        self._vis = torch.as_tensor(vis, dtype=torch.float32)
        if self._vis.dim() == 2:
            self._vis = self._vis.unsqueeze(
                -1
            )  # Add dimension for n_values if not present

        self.grad_cache = None

    def contain(self, pts):
        return self._meta.contain(pts)

    @classmethod
    def load(cls, cfg_or_fname: str):
        if isinstance(cfg_or_fname, dict):
            filepath = cfg_or_fname["photonlib"]["filepath"]
        elif isinstance(cfg_or_fname, str):
            filepath = cfg_or_fname
        else:
            raise ValueError(
                f"The argument of load function must be str or dict (received {cfg_or_fname} {type(cfg_or_fname)})"
            )

        meta = VoxelMeta.load(filepath)

        print(f"[PhotonLib] loading {filepath}")
        with h5py.File(filepath, "r") as f:
            vis = torch.as_tensor(f["vis"][:])
            eff = torch.as_tensor(f.get("eff", default=1.0))
        print("[PhotonLib] file loaded")

        return cls(meta, vis, eff)

    @property
    def meta(self):
        return self._meta

    @property
    def device(self):
        return self._vis.device

    def to(self, device=None):
        if device is None or self.device == torch.device(device):
            return self
        return TPhotonLib(self.meta, self.vis.to(device), self.eff.to(device))

    def visibility(self, x):
        """
        Returns the visibilities for all PMTs given the position(s) in x.

        Parameters
        ----------
        x : torch.Tensor
            A (or an array of) 3D point in the absolute coordinate

        Returns
        -------
        torch.Tensor
            An instance holding the visibilities for the position(s) x.
        """
        pos = x.unsqueeze(0) if x.dim() == 1 else x
        vis = torch.zeros(
            pos.shape[0],
            self.n_pmts,
            self.n_values,
            dtype=torch.float32,
            device=self.device,
        )
        mask = self.meta.contain(pos)
        vis[mask] = self.vis[self.meta.coord_to_voxel(pos[mask])]
        return vis.squeeze(0) if x.dim() == 1 else vis

    def gradx(self, x):
        pos = x.unsqueeze(0) if x.dim() == 1 else x
        grad = torch.zeros(
            pos.shape[0],
            self.n_pmts,
            self.n_values,
            dtype=torch.float32,
            device=self.device,
        )

        mask0 = self.meta.contain(pos)
        vox_ids = self.meta.coord_to_voxel(pos[mask0]) + 1
        mask1 = vox_ids >= (len(self.meta) - 1)
        vox_ids[mask1] = len(self.meta) - 1

        grad[mask0] = (
            self.vis[vox_ids] - self.vis[vox_ids - 1]
        ) / self.meta.voxel_size[0]

        return grad.squeeze(0) if x.dim() == 1 else grad

    def visibility2(self, x):
        pos = x.unsqueeze(0) if x.dim() == 1 else x
        return self._vis[self.meta.coord_to_voxel(pos)]

    def gradx2(self, x):
        pos = x.unsqueeze(0) if x.dim() == 1 else x

        if not hasattr(self, "grad"):
            self.grad = torch.zeros(
                len(self),
                self.n_pmts,
                self.n_values,
                dtype=torch.float32,
                device=self.device,
            )
            self.grad[:-1, :, :] = self._vis[1:] - self._vis[:-1]
            self.grad[-1, :, :] = self.grad[-2, :, :]

        vox_ids = self.meta.coord_to_voxel(pos) + 1
        return self.grad[vox_ids]

    @property
    def eff(self):
        return self._eff

    @property
    def vis(self):
        return self._vis

    def view(self, arr):
        shape = list(self.meta.shape.numpy()[::-1]) + [-1]
        new_order = list(range(len(shape)-1))[::-1] + [len(shape)-1]
        return arr.view(*shape).permute(*new_order)

    @property
    def vis_view(self):
        return self.view(self.vis)

    def __repr__(self):
        return f"{self.__class__} ({self.device})"

    def __len__(self):
        return len(self.vis)

    @property
    def n_pmts(self):
        return self.vis.shape[1]

    @property
    def n_values(self):
        return self.vis.shape[2]

    def __getitem__(self, vox_id):
        return self.vis[vox_id]

    def __call__(self, coords):
        return self.visibility(coords) * self.eff

    @staticmethod
    def save(outpath, vis, meta, eff=None):
        if isinstance(vis, torch.Tensor):
            vis = vis.cpu().detach().numpy()
        else:
            vis = np.asarray(vis)

        if vis.ndim == 5:  # Assuming (X, Y, Z, ..., n_value)
            
            shape = tuple(list(range(vis.ndim-1))[::-1] + [vis.ndim-1])
            vis = np.transpose(vis, shape).reshape(
                len(meta), -1, vis.shape[-1]
            )

        print("[PhotonLib] saving to", outpath)
        with h5py.File(outpath, "w") as f:
            f.create_dataset("numvox", data=meta.shape.cpu().detach().numpy())
            f.create_dataset("min", data=meta.ranges[:, 0].cpu().detach().numpy())
            f.create_dataset("max", data=meta.ranges[:, 1].cpu().detach().numpy())
            f.create_dataset("vis", data=vis, compression="gzip")

            if eff is not None:
                f.create_dataset("eff", data=eff)

        print("[PhotonLib] file saved")
        
        
