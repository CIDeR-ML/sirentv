from typing import List, Literal

import numpy as np
import torch
from slar.base import Siren
from torch import nn

from sirentv.models import MODELS
from sirentv.utils.misc import t0_mask
@MODELS.register_module()
class ParallelSiren(nn.Module):
    def __init__(self,
                 in_features: int = 6,
                 hidden_features: List[int] = [128, 256],
                 hidden_layers: List[int] = [2, 3],
                 out_features: list = [1, 1001],
                 outermost_linear: bool = True,
                 first_omega_0: float = 30.0,
                 hidden_omega_0: float = 30.0,
                 steepness_factor: float = 10.0,
                 use_CDF: bool = True,
                 xform_vis: dict = {},
                 ):
        super().__init__()

        hidden_features = (
            [hidden_features] if isinstance(hidden_features, int) else hidden_features
        )
        hidden_layers = (
            [hidden_layers] if isinstance(hidden_layers, int) else hidden_layers
        )
        assert isinstance(out_features, list) and len(out_features)==2, "Parallel network needs exactly list of 2 output features"
        self.out_features = out_features
        self._use_CDF = use_CDF

        # branch for voltage output
        self.v_net = Siren(
            in_features=in_features,
            out_features=out_features[0],
            hidden_features=hidden_features[0],
            hidden_layers=hidden_layers[0],
            outermost_linear=True,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )

        # branch for t0 and cdf outputs
        self.t0_cdf_net = Siren(
            in_features=in_features,
            out_features=out_features[1],
            hidden_features=hidden_features[1] if len(hidden_features)>1 else hidden_features[0],
            hidden_layers=hidden_layers[1] if len(hidden_layers)>1 else hidden_features[0],
            outermost_linear=True,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )

        self._steepness_factor = float(steepness_factor)
        self.hidden_omega_0 = hidden_omega_0

        self.init_weights()

    def forward(self, x):
        out_v = self.v_net(x)
        out_t0cdf = self.t0_cdf_net(x)
        out_t0, out_cdf = out_t0cdf[:, :, 1], out_t0cdf[:, :, 1:]
        #out_v = out_v - 8  # <-- initalize guess with 1e-8 offset
        n_ticks = out_cdf.shape[-1]
        t0 = torch.sigmoid(out_t0)*n_ticks  # t0 between 0 and 1000

        out_cdf = t0_mask(n_ticks, t0.unsqueeze(-1), out_cdf, self._steepness_factor, self._use_CDF)

        output = dict(
            v=out_v.squeeze(-1),
            t=out_cdf,
            t0=t0
        )

        return output

    def init_weights(self):
        """
        Siren initializes all first layer weights with a uniform distribution, and not the
        custom distribution often used for SIREN layers.
        """
        with torch.no_grad():
            for layer in self.v_net.net:
                if isinstance(layer, nn.Linear):
                    layer.weight.uniform_(
                        -np.sqrt(6 / layer.in_features) / self.hidden_omega_0,
                        np.sqrt(6 / layer.in_features) / self.hidden_omega_0,
                    )
                else:
                    layer.init_weights()

            for layer in self.t0_cdf_net.net:
                if isinstance(layer, nn.Linear):
                    layer.weight.uniform_(
                        -np.sqrt(6 / layer.in_features) / self.hidden_omega_0,
                        np.sqrt(6 / layer.in_features) / self.hidden_omega_0,
                    )
                else:
                    layer.init_weights()