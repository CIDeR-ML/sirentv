from __future__ import annotations

from photonlib import AABox
import torch
from sirentv import SirenTV

class MultiSirenTV(torch.nn.Module):
    def __init__(self, cfg: dict = None):
        super().__init__()

        self._meta = None
        self._model_v = []
        self._n_pmts_v = []
        self._aabox_array = []

        if cfg is None:
            return

        self.config = cfg.get("multivis")
        if isinstance(self.config, dict):
            if "ckpt_file" in self.config:
                filepath = self.config["ckpt_file"]
                print("[MultiVis] creating from checkpoint", filepath)
                with open(filepath, "rb") as f:
                    model_dict = torch.load(f, map_location="cpu")

                    self.load_model_dict(model_dict)

            else:
                print("[MultiVis] creating from a configuration dict...")
                self.configure(cfg)

    def to(self, device):
        self._meta = self._meta.to(device)
        super().to(device)
        return self

    def configure(self, cfg: dict):
        print("\n[MultiVis] configuring")
        self.config = dict(cfg)
        self.mvis_config = cfg["multivis"]
        weights = self.mvis_config.get("weights", [])
        weight_id = self.mvis_config.get("weight_id", [])
        input_scale = self.mvis_config.get("input_scale", [])
        aabox = self.mvis_config.get("aabox", [])
        if len(weights) and len(weight_id):
            # Validate config
            assert (
                len(weight_id) == len(input_scale) == len(aabox)
            ), f"weight_id ({len(weight_id)}), input_scale ({len(input_scale)}), aabox ({len(aabox)}) must be the same length."
            assert (
                max(weight_id) < len(weights)
            ), f"weight_id max value ({max(weight_id)}) must be less than {len(weights)}"

            for i, w in enumerate(weights):
                print(f"[MultiVis] adding {i+1}/{len(weights)} SirenTV instances...\n")
                self.add_model(SirenTV.load(w))

            aabox = torch.as_tensor(aabox)
            for i in range(len(weight_id)):
                m = AABox(aabox[i].T.reshape(3, 2))
                print("adding meta", i)
                self.add_meta(weight_id[i], m, input_scale[i])

        print("[MultiVis] configure finished\n")
        return

    def contain(self, pts):
        barray = torch.zeros(size=(len(pts),)).bool()
        for box in self._aabox_array:
            barray = barray | box.contain(pts)
        return barray

    @property
    def n_pmts(self):
        return sum(self._n_pmts_v)

    @property
    def meta(self):
        return self._meta

    @property
    def model_array(self):
        return self._model_v

    @property
    def device(self):
        """
        Access the device this model is on. This function assumes all parameters are on the same device.

        Returns
        -------
        torch.device
            The device ID where this model instance resides on
        """
        return next(self.model_array[0].parameters()).device

    def add_model(self, vis: SirenTVis):
        model_name = "model%d" % len(self._model_v)
        self.add_module(model_name, vis)
        self._model_v.append(getattr(self, model_name))

    def add_meta(self, model_idx: int, meta: AABox, input_scale: torch.Tensor = None):
        assert (
            model_idx < len(self.model_array)
        ), f"model_index given ({model_idx}) is invalid (must be <{len(self.model_array)}"

        if input_scale:
            input_scale = (
                torch.as_tensor(input_scale, dtype=torch.float32).clone().detach()
            )
        else:
            input_scale = torch.ones(size=len(meta.ranges), dtype=torch.float32)
        assert len(input_scale) == len(
            meta.ranges
        ), f"input_scale dimension must be {len(meta.ranges)}"

        # Append ranges
        if not hasattr(self, "children_meta"):
            self.register_buffer(
                "children_meta", meta.ranges.clone().detach()[None, :], persistent=False
            )
            self._aabox_array.append(AABox(self.children_meta[-1]))
        else:
            # make sure the given meta has no overlap with others
            for rs in self.children_meta:
                m = AABox(rs)
                if m.overlaps(meta):
                    raise ValueError(
                        f"The provided meta \n{meta}overlaps with one of registered meta \n{m}"
                    )
            self.children_meta = torch.cat([self.children_meta, meta.ranges[None, :]])
            print("adding to aabox array")
            self._aabox_array.append(AABox(self.children_meta[-1]))

        # Update own meta
        if self._meta is None:
            self._meta = meta
            self.register_buffer("meta_range", meta.ranges, persistent=False)
        else:
            self._meta.merge(meta)

        # Append the scaling factor
        if not hasattr(self, "input_scale"):
            self.register_buffer("input_scale", input_scale[None, :], persistent=False)
        else:
            self.input_scale = torch.cat([self.input_scale, input_scale[None, :]])

        # Append model id
        if not hasattr(self, "model_ids"):
            self.register_buffer(
                "model_ids", torch.as_tensor([model_idx]), persistent=False
            )
        else:
            self.model_ids = torch.cat([self.model_ids, torch.as_tensor([model_idx])])

        # PMT count
        self._n_pmts_v.append(self.model_array[model_idx].n_outs)

    def visibility2(self, x):
        if len(x.shape) == 1:
            x = x[None, :]
        device = x.device

        vis = torch.zeros(
            size=(x.shape[0], self.n_outs), dtype=torch.float32, device=self.device
        )

        for i, rs in enumerate(self.children_meta):
            m = AABox(rs)
            mask = m.contain(x)
            if mask.sum() < 1:
                continue
            model = self.model_array[self.model_ids[i]]
            # scale = self.input_scale[i]
            pos = m.norm_coord(x[mask]).to(self.device)
            pos *= self.input_scale[i]
            vis[mask, sum(self._n_pmts_v[:i]) : sum(self._n_pmts_v[: i + 1])] += (
                model._inv_xform_vis(model(pos))
            )

        return vis.to(device)

    def visibility(self, x):
        if len(x.shape) == 1:
            x = x[None, :]
        device = x.device
        x = x.to(self.device)

        vis = torch.zeros(
            size=(x.shape[0], self.n_pmts), dtype=torch.float32, device=self.device
        )
        vis_masks_pos = [[] for _ in range(max(self.model_ids) + 1)]
        vis_masks_pmt = [[] for _ in range(max(self.model_ids) + 1)]
        out_masks = [[] for _ in range(max(self.model_ids) + 1)]
        norm_x = [[] for _ in range(max(self.model_ids) + 1)]
        pos_ctr = [0 for _ in range(max(self.model_ids) + 1)]
        for i, rs in enumerate(self.children_meta):
            # m=AABox(rs)
            m = self._aabox_array[i]
            mask = m.contain(x)
            if mask.sum() < 1:
                continue

            model_idx = self.model_ids[i]

            norm_x[model_idx].append(m.norm_coord(x[mask]) * self.input_scale[i])
            vis_masks_pos[model_idx].append(mask)
            vis_masks_pmt[model_idx].append(
                [sum(self._n_pmts_v[:i]), sum(self._n_pmts_v[: i + 1])]
            )
            out_masks[model_idx].append(
                [pos_ctr[model_idx], pos_ctr[model_idx] + mask.sum().item()]
            )
            pos_ctr[model_idx] += mask.sum().item()

        for model_id, model in enumerate(self._model_v):
            if model_id >= len(norm_x):
                break
            if len(norm_x[model_id]) < 1:
                continue
            out = model._inv_xform_vis(
                model(torch.concat(norm_x[model_id]).to(self.device))
            )
            # out = model(torch.concat(norm_x[model_id]).to(self.device))
            # map back to vis array
            for meta_idx in range(len(vis_masks_pmt[model_id])):
                pos_mask = vis_masks_pos[model_id][meta_idx]
                pmt_start, pmt_end = vis_masks_pmt[model_id][meta_idx]
                out_start, out_end = out_masks[model_id][meta_idx]
                vis[pos_mask, pmt_start:pmt_end] += out[out_start:out_end, :]

        return vis.to(device)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x[None, :]
        device = x.device
        x = x.to(self.device)

        out = torch.zeros(
            size=(x.shape[0], self.n_pmts), dtype=torch.float32, device=self.device
        )

        for i, rs in enumerate(self.children_meta):
            # m = AABox(rs)
            m = self._aabox_array[i]
            mask = m.contain(x)
            if mask.sum() < 1:
                continue
            model = self.model_array[self.model_ids[i]]
            scale = self.input_scale[i]
            model.update_meta(meta=m, input_scale=scale)
            out[mask, sum(self._n_pmts_v[:i]) : sum(self._n_pmts_v[: i + 1])] += model(
                m.norm_coord(x[mask]).to(self.device)
            )

        return out.to(device)

    def model_dict(self, opt=None, sch=None, epoch=-1):
        # It's possible to save all underlying models with self.state_dict()
        # But WE DO NOT DO THIS: because there are individual SirenTVis attributes that cannot be retrieved.
        # Instead, we use each SirenTVis function to store/restore, and store this model specific parameters by hand.
        model_dict = dict(
            meta_range=self.meta.ranges,
            input_scale=self.input_scale,
            model_ids=self.model_ids,
            children_meta=self.children_meta,
        )
        for i, model in enumerate(self._model_v):
            if model is None:
                continue
            print("[MultiVis] Saving model", i)
            model_dict["model%d" % i] = model.model_dict()
        if opt:
            model_dict["optimizer"] = opt.state_dict()
        if sch:
            model_dict["scheduler"] = sch.state_dict()
        if epoch >= 0:
            model_dict["epoch"] = epoch
        return model_dict

    def save_state(self, filename, opt=None, sch=None, epoch=-1):
        """
        Stores the network model and optimizer (and some hyper-) parameters to a binary file.


        Parameters
        ----------
        filename : str
            The name of checkpoint file to be stored.
        opt : torch.optim.Optimizer
            The optimizer instance to store the optimizer state in the same file.
        epoch : float
            The epoch count of training.
        """

        print("[MultiVis] saving the state ", filename)
        torch.save(self.model_dict(opt, sch, epoch), filename)

    def load_model_dict(self, model_dict):
        """
        Loads the network model and optimizer (and some hyper-) parameters from a dictionary
        Parameters
        ----------
        model_dict : dict
            Contains all model parameters necessary to re-instantiate the mode at the checkpoint.

        """
        from photonlib import AABox

        print("[MultiVis] loading model_dict")
        self.register_buffer("meta_range", model_dict["meta_range"], persistent=False)
        self.register_buffer("input_scale", model_dict["input_scale"], persistent=False)
        self.register_buffer("model_ids", model_dict["model_ids"], persistent=False)
        self.register_buffer(
            "children_meta", model_dict["children_meta"], persistent=False
        )
        self._meta = AABox(self.meta_range)
        self._aabox_array = []
        for r in self.children_meta:
            self._aabox_array.append(AABox(r))

        self._model_v = [None] * len(self.model_ids)
        for mid in self.model_ids:
            name = "model%d" % mid
            if not name in model_dict:
                raise KeyError(f"Model {name} not found !")
            if self._model_v[mid] is None:
                print(f"[MultiVis] Creating SirenTVis as {name}")
                model = SirenTVis.create_from_model_dict(model_dict[name])
                self.add_module(name, model)
                self._model_v[mid] = getattr(self, name)

            self._n_pmts_v.append(self._model_v[mid].n_outs)
        print("[MultiVis] loading finished\n")

    def load_state(self, model_path):
        """
        Loads the network model and optimizer (and some hyper-) parameters from a binary file.

        Parameters
        ----------
        model_path : str
            The checkpoint file name from which parameter values are loaded.

        """

        print("[MultiVis] loading from checkpoint", model_path)
        with open(model_path, "rb") as f:
            model_dict = torch.load(f, map_location="cpu")

            self.load_model_dict(model_dict)

    @classmethod
    def load(cls, cfg_or_fname: str | dict):
        """
        Constructor method that can take either a config dictionary or the data file path

        Parameters
        ----------
        cfg_or_fname : str
            If string type, it is interpreted as a path to a photon library data file.
            If dictionary type, it is interpreted as a configuration.
        """

        if isinstance(cfg_or_fname, dict):
            if not "multivis" in cfg_or_fname:
                raise KeyError("The configuration dictionary must contain multivis")
            if "ckpt_file" in cfg_or_fname["multivis"]:
                filepath = cfg_or_fname["multivis"]["ckpt_file"]
            else:
                print("[MultiVis] creating from a configuration dict...")
                return cls(cfg_or_fname)
        elif isinstance(cfg_or_fname, str):
            filepath = cfg_or_fname
        else:
            raise ValueError(
                f"The argument of load function must be str or dict (received {cfg_or_fname} {type(cfg_or_fname)})"
            )

        print("[MultiVis] creating from checkpoint", filepath)
        with open(filepath, "rb") as f:
            model_dict = torch.load(f, map_location="cpu")

            out = cls()

            out.load_model_dict(model_dict)

            return out
