import numpy as np
import torch
from slar.base import Siren
from torch import nn

from sirentv.models import MODELS


@MODELS.register_module()
class PcaSiren(nn.Module):
    """Single SIREN network outputting PCA coefficients + visibility + log_t0."""

    def __init__(
        self,
        in_features=6,
        hidden_features=512,
        hidden_layers=3,
        n_components=50,
        outermost_linear=True,
        first_omega_0=30.0,
        hidden_omega_0=30.0,
        **kwargs,
    ):
        super().__init__()
        out_features = n_components + 2  # coeffs + vis + log_t0
        self.n_components = n_components
        self.out_features = [2, n_components]  # for SirenTV wrapper compatibility

        self.net = Siren(
            in_features,
            hidden_features,
            hidden_layers,
            out_features,
            outermost_linear,
            first_omega_0,
            hidden_omega_0,
        )

    def forward(self, x, *args, **kwargs):
        out = self.net(x)              # (B, N_pmt, n_components+2)
        vis = out[..., 0]              # (B, N_pmt)
        log_t0 = out[..., 1]          # (B, N_pmt)
        coeffs = out[..., 2:]         # (B, N_pmt, n_components)
        return dict(v=vis, t=coeffs, t0=log_t0)
