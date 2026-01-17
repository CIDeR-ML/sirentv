import math
from typing import List, Literal, Union

import numpy as np
import torch
from photonlib import AABox
from slar.base import Siren
from slar.transform import partial_xform_vis
from torch import nn

from sirentv.models import MODELS
from sirentv.utils.misc import t0_mask
@MODELS.register_module()
class BranchedSiren(nn.Module):
    def __init__(self,
                 in_features: int = 6,
                 hidden_features: List[int] = [256, 256, 1024],
                 hidden_layers: List[int] = [2, 3, 3],
                 out_features: list = [1, 1001],
                 outermost_linear: bool = False,
                 first_omega_0: float = 30.0,
                 hidden_omega_0: float = 30.0,
                 steepness_factor: float = 10.0,
                 use_CDF: bool = True,
                 use_t0: bool = True,
                 xform_vis: dict ={},
    ):
        super().__init__()

        hidden_features = (
            [hidden_features] if isinstance(hidden_features, int) else hidden_features
        )
        hidden_layers = (
            [hidden_layers] if isinstance(hidden_layers, int) else hidden_layers
        )
        assert isinstance(out_features, list) and len(out_features)==2, "WaveformSiren needs exactly list of 2 output features"

        self._use_t0 = use_t0
        
        # adjust waveform decoder output size based on whether t0 is used
        # if use_t0=False, only output CDF (1000), not t0+CDF (1001)
        waveform_out_features = out_features[1] if use_t0 else out_features[1] - 1

        print("=" * 20, "visibility encoder", "=" * 20)
        self.encoder = Siren(
            in_features=in_features,
            hidden_features=hidden_features[0],
            hidden_layers=hidden_layers[0] - 1,
            out_features=hidden_features[0],
            outermost_linear=False,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )
        print("=" * 20, "Visibility decoder", "=" * 20)
        self.vis_decoder = Siren(
            in_features=hidden_features[0],
            hidden_features=hidden_features[1],
            hidden_layers=hidden_layers[1] - 1,
            out_features=out_features[0],
            outermost_linear=outermost_linear,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )

        print("=" * 20, "Waveform decoder", "=" * 20)
        self.waveform_decoder = Siren(
            in_features=hidden_features[0],
            hidden_features=hidden_features[2],
            hidden_layers=hidden_layers[2] - 1,
            out_features=waveform_out_features,
            outermost_linear=outermost_linear,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )

        self.hidden_omega_0 = hidden_omega_0
        self.check_outputs()

        self.init_weights()
        self.out_features = out_features

        self._steepness_factor = float(steepness_factor)
        self._use_CDF = use_CDF

    def check_outputs(self):
        assert (
            self.encoder.net[-1].linear.out_features
            == self.vis_decoder.net[0].linear.in_features
        )
        assert (
            self.encoder.net[-1].linear.out_features
            == self.waveform_decoder.net[0].linear.in_features
        )

    def forward(self, x):
        x = self.encoder(x)
        out_v = self.vis_decoder(x)
        
        if self._use_t0:
            out_t0cdf = self.waveform_decoder(x)
            out_t0, out_cdf = out_t0cdf[:, :, 0], out_t0cdf[:, :, 1:]
            n_ticks = out_cdf.shape[-1]
            t0 = torch.sigmoid(out_t0) * n_ticks
            out_cdf = t0_mask(n_ticks, t0.unsqueeze(-1), out_cdf, self._steepness_factor, self._use_CDF)
        else:
            out_cdf = self.waveform_decoder(x) # (B, N_pmt, N_time)
            out_cdf = out_cdf.softmax(dim=-1) # (B, N_pmt, N_time)
            if self._use_CDF:
                out_cdf = out_cdf.cumsum(dim=-1)
            t0 = None

        output = dict(
            v=out_v.squeeze(-1),
            t=out_cdf,
            t0=t0,
        )

        return output

    def init_weights(self):
        """
        Siren initializes all first layer weights with a uniform distribution, and not the
        custom distribution often used for SIREN layers.
        """
        with torch.no_grad():
            for layer in self.encoder.net:
                if isinstance(layer, nn.Linear):
                    layer.weight.uniform_(
                        -np.sqrt(6 / layer.in_features) / self.hidden_omega_0,
                        np.sqrt(6 / layer.in_features) / self.hidden_omega_0,
                    )
                else:
                    layer.init_weights()

            for decoder in [self.vis_decoder, self.waveform_decoder]:
                for layer in decoder.net:
                    if isinstance(layer, nn.Linear):
                        layer.weight.uniform_(
                            -np.sqrt(6 / layer.in_features) / self.hidden_omega_0,
                            np.sqrt(6 / layer.in_features) / self.hidden_omega_0,
                        )
                    else:
                        layer.is_first = False
                        layer.init_weights()

            assert all(not layer.is_first for layer in self.encoder.net[1:])

            assert all(
                not layer.is_first
                for layer in self.waveform_decoder.net
                if not isinstance(layer, nn.Linear)
            )