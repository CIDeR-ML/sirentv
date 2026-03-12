from typing import List

from slar.base import Siren
from torch import nn

from sirentv.models import MODELS


@MODELS.register_module()
class DualPcaSiren(nn.Module):
    """Two independent SIREN branches: one for PCA coefficients, one for vis + log_t0."""

    def __init__(
        self,
        in_features: int = 6,
        hidden_features: List[int] | int = [256, 512],
        hidden_layers: List[int] | int = [3, 3],
        n_components: int = 50,
        outermost_linear: bool = True,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        **kwargs,
    ):
        super().__init__()
        if isinstance(hidden_features, int):
            hidden_features = [hidden_features, hidden_features]
        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers, hidden_layers]

        self.n_components = n_components
        self.out_features = [2, n_components]

        self.vt0_net = Siren(
            in_features,
            hidden_features[0],
            hidden_layers[0],
            2,  # vis + log_t0
            outermost_linear,
            first_omega_0,
            hidden_omega_0,
        )

        self.coeff_net = Siren(
            in_features,
            hidden_features[1] if len(hidden_features) > 1 else hidden_features[0],
            hidden_layers[1] if len(hidden_layers) > 1 else hidden_layers[0],
            n_components,
            outermost_linear,
            first_omega_0,
            hidden_omega_0,
        )

    def forward(self, x, *args, **kwargs):
        vt0 = self.vt0_net(x)        # (B, N_pmt, 2)
        coeffs = self.coeff_net(x)    # (B, N_pmt, n_components)

        vis = vt0[..., 0]             # (B, N_pmt)
        log_t0 = vt0[..., 1]         # (B, N_pmt)
        return dict(v=vis, t=coeffs, t0=log_t0)
