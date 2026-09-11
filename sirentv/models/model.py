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
from sirentv.data.compressed import CompressedPLib


class SirenTV(nn.Module):
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
            #return

        # Create meta
        if meta is not None:
            self._meta = meta
        elif "compressed_plib" in cfg:
            cplib_cfg = cfg["compressed_plib"]
            # only .meta/.pmt_pos are used below, both read unconditionally at construction --
            # lazy=True here never touches vis/t0/coeffs, so there's no reason this should ever
            # eagerly pull the full (tens-of-GB) dataset into memory just to build positional
            # metadata (this used to hardcode lazy=False and could OOM-kill a modest-memory
            # process, e.g. a Jupyter kernel, for no benefit)
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

        # Transform functions
        self._xform_vis, self._inv_xform_vis = partial_xform_vis(self.config_xform)

        # Extensions for visibility model
        self._init_output_scale(self.config_model)
        self.tick_size = self.config_model.get("tick_size", 0.1) # ns

        self.n_pmts: int = self.config_data.get("n_pmt", 81)

        anneal_cfg = self.config_model.get("anneal", {})
        self.anneal_enabled = anneal_cfg.get("enabled", False)
        self.current_tau = 1.0
        if self.anneal_enabled:
            self.tau_start = anneal_cfg.get("tau_start", 1.0)
            self.tau_end = anneal_cfg.get("tau_end", 0.01)
            self.current_tau = self.tau_start

    def anneal_temperature(self, epoch, max_epochs):
        """Call this at end of each epoch if annealing is enabled"""
        if self.anneal_enabled:
            self.current_tau = self.tau_start * (self.tau_end / self.tau_start) ** (epoch / max_epochs)

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

    def forward(self, x, return_gradients=False, grad_keys=None, grad_aggregate=False, grad_create_graph=True,
                grad_pmt_ids=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input in unnormalized coordinates.
        return_gradients: bool
            Whether to compute the visibility gradient in the forward (legacy path, used only
            by BranchedSiren's own internal analytical-gradient computation).
        grad_keys : list[str], optional
            For each key in this list, additionally compute a `{key}_grad_frob` output: a
            genuinely differentiable (create_graph=True by default) SQUARED Frobenius-norm
            gradient magnitude of out[key] w.r.t. physical (unnormalized) position -- see
            compute_grad_frob_hutchinson (squared, not the sqrt'd norm, so the estimator stays
            unbiased at a single projection -- see WeightedGradFrobLoss's docstring for why).
        grad_aggregate : bool
            False (default): per-(voxel, PMT) granularity via compute_grad_frob_hutchinson --
            O(n_pmts) backward passes per call, independent of channel count K, but n_pmts
            itself (e.g. 81) can still OOM on some hardware. True: fall back to
            compute_grad_frob_hutchinson_aggregate -- a single visibility-weighted aggregate
            over all PMTs at once (O(1) backward passes total), trading away per-PMT
            granularity for a much cheaper estimate. `{key}_grad_frob` becomes (n_valid,)
            instead of (n_valid, n_pmts) in this mode.
        grad_create_graph : bool
            Passed straight through to compute_grad_frob_hutchinson[_aggregate]. True
            (default) is required during training, where `{key}_grad_frob` feeds into a loss
            that itself gets backpropagated. Pass False for a read-only use (e.g. a notebook
            diagnostic that only wants the VALUES) -- PyTorch retains substantially more graph
            structure per backward pass when this is True, so leaving it True when nothing
            downstream differentiates through the result again is a real, avoidable memory
            cost, particularly with grad_aggregate=False's O(n_pmts) backward passes per call.
        grad_pmt_ids : list[int], optional
            Only used when grad_aggregate=False. Restricts compute_grad_frob_hutchinson's
            O(n_pmts) backward-pass loop to just these PMT indices -- e.g. a diagnostic that
            only ever reads off one fixed PMT has no reason to pay for the other n_pmts-1
            backward passes. None (default) computes every PMT, as training needs all of them.
            `{key}_grad_frob` becomes (n_valid, len(grad_pmt_ids)) instead of (n_valid, n_pmts)
            when set -- indexed in the SAME order as grad_pmt_ids, not by absolute PMT index.
        return_pdf : bool
            If True, return the PDF of the waveform. If False, return the CDF.

        Returns
        -------
        out : dict
            Dictionary containing the PDF/CDF of the waveform and the visibility.
            The keys are "t" and "v".
        """
        #device = x.device
        pos = x.unsqueeze(0) if x.dim() == 1 else x
        mask = self.meta.contain(pos).to(self.device)
        pos_masked = pos[mask]
        if grad_keys:
            # fresh leaf, detached from whatever graph state x had -- position is a fixed model
            # input, not a trainable parameter, so there's nothing upstream worth preserving;
            # we only need d(output)/d(pos_masked) itself, not further backprop past it
            pos_masked = pos_masked.detach().requires_grad_(True)
        norm_pos = self.meta.norm_coord(pos_masked).to(self.device)
        if norm_pos.dim() == 1:
            norm_pos = norm_pos.unsqueeze(0)
        norm_pos = norm_pos.unsqueeze(1).expand(-1, self.n_pmts, -1)
        input_to_net = norm_pos
        if self._load_pos:
            norm_pmt_tile = self.norm_pmt_coords.to(self.device).unsqueeze(0).expand_as(norm_pos)
            input_to_net = torch.cat([norm_pos, norm_pmt_tile], dim=-1)

        out = self.model(input_to_net, self.current_tau, return_gradients)

        if grad_keys:
            if grad_aggregate:
                from sirentv.utils.misc import compute_grad_frob_hutchinson_aggregate
                # predicted visibility (this forward pass's own, detached inside the estimator)
                # is the only visibility signal available here -- target visibility isn't
                # passed into forward() at all, so the target-side aggregate (computed offline
                # by the dataset) uses TRUE visibility instead. A known, accepted asymmetry for
                # this auxiliary loss -- see discussion at the call site in train.py/config.
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
                        pmt_ids=grad_pmt_ids,
                    )

        result = {"correct_mask": mask}
        for key, val in out.items():
            # grad_mags default to 1.0 (unit gradient) for masked positions
            default_val = 1.0 if key == "grad_mags_transformed" else 0.0
            buf = torch.full(
                (pos.shape[0], *val.shape[1:]),
                default_val,
                dtype=torch.float32,
                device=self.device,
            )
            buf[mask] = val.to(device=self.device, dtype=torch.float32)
            result[key] = buf

        return result

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

        if v.dim() != t.dim():
            v = v.unsqueeze(-1)
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

    def freeze_all(self):
        """Unfreeze all parameters in the network"""
        for param in self.model.parameters():
            param.requires_grad = False
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