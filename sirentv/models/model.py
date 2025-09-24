from typing import Union, Literal

import os
import numpy as np
import torch
import h5py
import torch.nn as nn
from photonlib import AABox
from slar.transform import partial_xform_vis

from sirentv.models.builder import build_model
from sirentv.utils.transform import cdf_to_pdf, pdf_to_cdf


class SirenTV(nn.Module):
    def __init__(self, cfg: dict, meta=None):
        super().__init__()
        self.config_model = cfg["model"]
        self.config_loader = cfg["data"]["loader"]
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
                model_dict = torch.load(f, map_location="cpu")
                self.load_model_dict(model_dict)
            return

        # Create meta
        if meta is not None:
            self._meta = meta
        elif "photonlib" in cfg:
            self._meta = AABox.load(cfg["photonlib"]["filepath"])

        with h5py.File(cfg["photonlib"]["filepath"], 'r') as file:
            self.norm_pmt_coords = torch.tensor(file['pmt_norm_pos'][:], dtype=torch.float32)
        # Transform functions
        self._xform_vis, self._inv_xform_vis = partial_xform_vis(self.config_xform)

        # Extensions for visibility model
        self._init_output_scale(self.config_model)
        self._do_hardsigmoid = self.config_model.get("hardsigmoid", False)
        self.tick_size = self.config_model.get("tick_size", 0.1) # ns

        self.n_pmts = len(self.norm_pmt_coords)
        self.batch_size = self.config_loader.get("batch_size", 1024)
        self.norm_pmt_tile = self.norm_pmt_coords.unsqueeze(0).expand(self.batch_size, self.n_pmts, 3)


    def to(self, device):
        self._meta.to(device)
        return super().to(device)

    def contain(self, pts):
        return self.meta.contain(pts)

    @property
    def meta(self):
        return self._meta

    @property
    def device(self):
        return next(self.parameters()).device

    def update_meta(self, ranges: torch.Tensor):
        self._meta.update(ranges)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input in unnormalized coordinates.
        return_pdf : bool
            If True, return the PDF of the waveform. If False, return the CDF.

        Returns
        -------
        out : dict
            Dictionary containing the PDF/CDF of the waveform and the visibility.
            The keys are "t" and "v".
        """
        #device = x.device
        x = x.to(self.device)
        pos = x.unsqueeze(0) if x.dim() == 1 else x
        assert len(pos) == self.batch_size, "Loader batch size not consistent with config"
        mask = self.meta.contain(pos).to(self.device)
        norm_pos = torch.tile(self.meta.norm_coord(pos[mask]).unsqueeze(1), (1, self.n_pmts, 1)).to(self.device)
        input_to_net = torch.cat([norm_pos, self.norm_pmt_tile.to(self.device)], dim=-1).to(self.device)
        out = self.model(input_to_net)#.to(device)

        v = torch.zeros(
            pos.shape[0], out['v'].shape[-1], dtype=torch.float32, device=self.device
        )
        v[mask] = out['v'].to(self.device)

        t = torch.zeros(
            pos.shape[0], *out['t'].shape[1:], dtype=torch.float32, device=self.device
        )
        t[mask] = out['t'].to(self.device)
        return {"t": t, "v": v, "correct_mask": mask}

    def visibility(self, x, return_type: Literal["pdf", "cdf"] = "pdf"):
        out = self.forward(x)
        t = out['t']
        v = out['v']
        mask = out['correct_mask']
        if self.mode == "cdf" and return_type == "pdf":
            # if cdf is returned, then it's in linear domain.
            # we just return pdf via diff/tick size.
            t = cdf_to_pdf(t, self.tick_size)
        elif self.mode == "pdf":
            # if pdf is returned by model, it's NOT in log domain
            # so we DO NOT need to convert it to linear domain
            # t[mask] = self._inv_xform_vis(t[mask])
            if return_type == "cdf":
                t = pdf_to_cdf(t)

        # TODO: we probably shouldn't use same transform rules
        # for both v and t as t << v.
        v[mask] = self._inv_xform_vis(v[mask]) # (B, N_pmt)
        return v.expand_as(t) * t # (B, N_pmt, N_time)

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
        self._do_hardsigmoid = self.config_model.get("hardsigmoid", False)
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
        """Number of output features separated by overall PMT visibility and waveform must be set by subclasses
        
        Example:
            out_features = [48, 4800] # 48 PMT visibility and 48x100 waveform features
        """
        if not hasattr(self, "_out_features"):
            raise NotImplementedError("Number of output features is not set in the model")
        return self._out_features

    @out_features.setter
    def out_features(self, value):
        self._out_features = value

    @property
    def n_outs(self):
        return sum(self.out_features) if isinstance(self.out_features, (list, tuple)) else self.out_features

    def unfreeze_all(self):
        """Unfreeze all parameters in the network"""
        for param in self.model.parameters():
            param.requires_grad = True
    def get_trainable_params(self):
        """Return only the parameters that require gradients"""
        return filter(lambda p: p.requires_grad, self.model.parameters())

    """
    def freeze_all_but_(self, part: Literal["timing", "visibility"] = "timing"):
        self.unfreeze_all()
        self.encoder.requires_grad_(False)
        if decoder == "timing":
            self.vis_decoder.requires_grad_(False)
        elif decoder == "visibility":
            self.waveform_decoder.requires_grad_(False)
        else:
            raise ValueError(f"Invalid decoder: {decoder}")
    """

    def print_trainable_params(self):
        """Print the names of trainable parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"Trainable: {name}")
            else:
                print(f"Frozen: {name}")

    def __repr__(self):
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        memory_mb = n_params * 4 / (1024 * 1024)  # assume fp32
        return f"{n_params:,} trainable parameters\n{memory_mb:2f} MB\n{super().__repr__()}"