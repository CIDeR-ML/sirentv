from typing import Union

import numpy as np
import torch
import h5py
import torch.nn as nn
from photonlib import AABox
from slar.transform import partial_xform_vis

from sirentv.models.builder import build_model
from sirentv.data.compressed import CompressedPLib


class SirenTV(nn.Module):
    """Wraps a registry network: normalizes positions, tiles PMTs, masks out-of-volume points."""

    def __init__(self, cfg: dict, meta=None, weights_only: bool = False):
        super().__init__()
        self.config_model = cfg["model"]
        self.config_data = cfg["data"]
        self.config_loader = cfg["data"]["loader"]
        self._load_pos = self.config_loader.get("load_pos", True)

        self.config_xform = cfg.get("transform_vis", None)
        if self.config_xform is None:
            print("[SirenTV] transform_vis is not set, using default values")
            self.config_xform = {}
        self.config_model['network']['xform_vis'] = self.config_xform

        self.mode = self.config_model.get("mode", "pdf")
        self._use_cdf = (self.mode.lower() == "cdf")
        model_config = self.config_model["network"]
        model_config.update({
            "use_CDF": self._use_cdf
        })
        self.model = build_model(model_config)
        self.out_features = self.model.out_features
        ckpt_file = self.config_model.get("ckpt_file")
        if ckpt_file:
            print("[SirenTV] loading model_dict from checkpoint", ckpt_file)
            with open(ckpt_file, "rb") as f:
                model_dict = torch.load(f, map_location="cpu", weights_only=weights_only)
                self.load_model_dict(model_dict)

        if meta is not None:
            self._meta = meta
        elif "compressed_plib" in cfg:
            cplib_cfg = cfg["compressed_plib"]
            # lazy: only meta/pmt_pos are needed, never the (tens-of-GB) data
            cplib = CompressedPLib.load(
                cplib_cfg["filepath"],
                lazy=True,
                n_components=cplib_cfg.get("n_components"),
            )
            self._meta = cplib.meta
            if self._load_pos:
                pmt_pos = torch.tensor(cplib.pmt_pos, dtype=torch.float32)
                self.pmt_coords = pmt_pos
                self.norm_pmt_coords = cplib.meta.norm_coord(pmt_pos)
            del cplib
        elif "photonlib" in cfg:
            self._meta = AABox.load(cfg["photonlib"]["filepath"])
            if self._load_pos:
                with h5py.File(cfg["photonlib"]["filepath"], 'r') as file:
                    self.pmt_coords = torch.tensor(file['pmt_pos'][:], dtype=torch.float32)
                    self.norm_pmt_coords = torch.tensor(file['pmt_norm_pos'][:], dtype=torch.float32)

        self._xform_vis, self._inv_xform_vis = partial_xform_vis(self.config_xform)
        self._init_output_scale(self.config_model)
        self.n_pmts: int = self.config_data.get("n_pmt", 81)

    def to(self, device):
        self._meta.to(device)
        return super().to(device)

    def contain(self, pts):
        return self.meta.contain(pts)

    @property
    def meta(self):
        return self._meta

    @property
    def load_pos(self):
        return self._load_pos

    @property
    def device(self):
        return next(self.parameters()).device

    def update_meta(self, ranges: torch.Tensor):
        self._meta.update(ranges)

    def forward(self, x, grad_keys=None, grad_aggregate=False, grad_create_graph=True,
                grad_pmt_ids=None, grad_n_projections=1):
        """x: (B, 3) unnormalized positions -> dict of (B, n_pmt, ...) outputs plus "correct_mask".

        Points outside the volume are masked out and read 0.
        grad_keys: also return `{key}_grad_frob`, the squared Frobenius norm of d out[key] / d x:
            grad_aggregate=False: per (voxel, PMT), compute_grad_frob_hutchinson; grad_pmt_ids
                restricts its per-PMT backward loop (columns follow grad_pmt_ids order; the
                forward still runs every PMT) and grad_n_projections sets its projections.
            grad_aggregate=True: one visibility-weighted value per voxel,
                compute_grad_frob_hutchinson_aggregate (O(1) backward passes).
        grad_create_graph: keep True for training; False for read-only diagnostics saves memory.
        """
        pos = x.unsqueeze(0) if x.dim() == 1 else x
        mask = self.meta.contain(pos).to(self.device)
        pos_masked = pos[mask]
        if grad_keys:
            # fresh leaf: only d(output)/d(position) is needed
            pos_masked = pos_masked.detach().requires_grad_(True)
        norm_pos = self.meta.norm_coord(pos_masked).to(self.device)
        if norm_pos.dim() == 1:
            norm_pos = norm_pos.unsqueeze(0)
        norm_pos = norm_pos.unsqueeze(1).expand(-1, self.n_pmts, -1)
        input_to_net = norm_pos
        if self._load_pos:
            norm_pmt_tile = self.norm_pmt_coords.to(self.device).unsqueeze(0).expand_as(norm_pos)
            input_to_net = torch.cat([norm_pos, norm_pmt_tile], dim=-1)

        out = self.model(input_to_net)

        if grad_keys:
            if grad_aggregate:
                from sirentv.utils.misc import compute_grad_frob_hutchinson_aggregate
                # weighted by the PREDICTED visibility (the target side uses the true one)
                vis_weight = self._inv_xform_vis(out["v"])
                for key in grad_keys:
                    out[f"{key}_grad_frob"] = compute_grad_frob_hutchinson_aggregate(
                        pos_masked, out[key], vis_weight, create_graph=grad_create_graph,
                    )
            else:
                from sirentv.utils.misc import compute_grad_frob_hutchinson
                for key in grad_keys:
                    out[f"{key}_grad_frob"] = compute_grad_frob_hutchinson(
                        pos_masked, out[key], self.n_pmts, create_graph=grad_create_graph,
                        pmt_ids=grad_pmt_ids, n_projections=grad_n_projections,
                    )

        result = {"correct_mask": mask}
        for key, val in out.items():
            buf = torch.full(
                (pos.shape[0], *val.shape[1:]),
                0.0,
                dtype=torch.float32,
                device=self.device,
            )
            buf[mask] = val.to(device=self.device, dtype=torch.float32)
            result[key] = buf

        return result

    def model_dict(self, opt=None, sch=None, epoch=-1, scaler=None):
        model_dict = {
            "state_dict": self.state_dict(),
            "xform_cfg": self.config_xform,
            "model_cfg": self.config_model,
            "aabox_ranges": self._meta.ranges,
        }
        if opt:
            model_dict["optimizer"] = opt.state_dict()
        if sch:
            model_dict["scheduler"] = sch.state_dict()
        if epoch >= 0:
            model_dict["epoch"] = epoch
        if scaler is not None:
            model_dict["amp_scaler"] = scaler.state_dict()
        return model_dict

    def save_state(self, filename, opt=None, sch=None, epoch=-1, scaler=None):
        print("[SirenTV] saving model_dict ", filename)
        torch.save(self.model_dict(opt, sch, epoch, scaler), filename)
        print("[SirenTV] saving finished")

    def load_model_dict(self, model_dict):
        print("[SirenTV] loading model_dict")

        self.config_model = model_dict.get("model_cfg")
        self.config_xform = model_dict.get("xform_cfg")
        if self.config_model is None:
            raise KeyError('The model dictionary is lacking the "model_cfg" data')

        self._init_output_scale(self.config_model)
        self._xform_vis, self._inv_xform_vis = partial_xform_vis(self.config_xform)

        self._meta = AABox(model_dict["aabox_ranges"])

        state_dict = model_dict["state_dict"]
        if "input_scale" in state_dict:
            state_dict.pop("input_scale")
        if "scale" in model_dict.keys():
            state_dict["output_scale"] = model_dict["scale"]

        self.load_state_dict(state_dict)

        print("[SirenTV] loading finished\n")

    @classmethod
    def load(cls, cfg_or_fname: Union[str, dict]):
        if isinstance(cfg_or_fname, dict):
            if "model" not in cfg_or_fname:
                raise KeyError("The configuration dictionary must contain model")
            if "ckpt_file" in cfg_or_fname["model"]:
                filepath = cfg_or_fname["model"]["ckpt_file"]
            else:
                print("[SirenTV] creating from a configuration dict...")
                return cls(cfg_or_fname)
        elif isinstance(cfg_or_fname, str):
            filepath = cfg_or_fname
        else:
            raise ValueError(
                f"The argument of load function must be str or dict (received {cfg_or_fname} {type(cfg_or_fname)})"
            )

        print("[SirenTV] creating from checkpoint", filepath)
        with open(filepath, "rb") as f:
            model_dict = torch.load(f, map_location="cpu")
            return cls.create_from_model_dict(model_dict)

    @classmethod
    def create_from_model_dict(cls, model_dict):
        cfg = {
            "model": model_dict["model_cfg"],
            "transform_vis": model_dict["xform_cfg"],
        }

        if "ckpt_file" in cfg["model"]:
            cfg["model"].pop("ckpt_file")

        net = cls(cfg)
        net.load_model_dict(model_dict)
        return net

    def _init_output_scale(self, siren_cfg):
        scale_cfg = siren_cfg.get("output_scale", {})
        init = scale_cfg.get("init")

        if init is None:
            output_scale = np.ones(self.n_outs)
        elif isinstance(init, str):
            output_scale = np.load(init)
        else:
            output_scale = np.asarray(init)

        assert len(output_scale) == self.n_outs, "len(output_scale) != out_features"

        output_scale = torch.tensor(np.nan_to_num(output_scale), dtype=torch.float32)

        if scale_cfg.get("fix", True):
            self.register_buffer("output_scale", output_scale, persistent=True)
        else:
            self.register_parameter("output_scale", torch.nn.Parameter(output_scale))

    @property
    def out_features(self):
        """Output widths, e.g. [v + t0, n_components]; set by the network."""
        if not hasattr(self, "_out_features"):
            raise NotImplementedError("Number of output features is not set in the model")
        return self._out_features

    @out_features.setter
    def out_features(self, value):
        self._out_features = value

    @property
    def n_outs(self):
        return sum(self.out_features) if isinstance(self.out_features, (list, tuple)) else self.out_features

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def get_trainable_params(self):
        return filter(lambda p: p.requires_grad, self.model.parameters())

    def print_trainable_params(self):
        for name, param in self.model.named_parameters():
            print(f"{'Trainable' if param.requires_grad else 'Frozen'}: {name}")

    def __repr__(self):
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        memory_mb = n_params * 4 / (1024 * 1024)  # assume fp32
        return f"{n_params:,} trainable parameters\n{memory_mb:2f} MB\n{super().__repr__()}"
