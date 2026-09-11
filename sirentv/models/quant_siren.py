import torch
from slar.base import Siren
from torch import nn

from sirentv.models import MODELS


@MODELS.register_module()
class QuantileSiren(nn.Module):
    """Single SIREN network outputting raw quantile values + visibility + log_t0."""

    def __init__(
        self,
        in_features=6,
        hidden_features=512,
        hidden_layers=3,
        n_quantile=256,
        outermost_linear=True,
        first_omega_0=30.0,
        hidden_omega_0=30.0,
        **kwargs,
    ):
        super().__init__()
        out_features = n_quantile + 2  # quantiles + vis + log_t0
        self.n_quantile = n_quantile
        self.out_features = [2, n_quantile]  # for SirenTV wrapper compatibility

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
        out = self.net(x)              # (B, N_pmt, n_quantile+2)
        vis = out[..., 0]              # (B, N_pmt)
        log_t0 = out[..., 1]          # (B, N_pmt)
        quantiles = out[..., 2:]      # (B, N_pmt, n_quantile)
        return dict(v=vis, quantiles=quantiles, t0=log_t0)
