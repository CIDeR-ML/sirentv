from __future__ import annotations

from sirentv.models import MODELS
from sirentv.models.multibranch import MultiBranchSiren


@MODELS.register_module()
class QuantileSiren(MultiBranchSiren):
    """Single SIREN trunk outputting raw quantile values + visibility + log_t0.

    Thin wrapper over MultiBranchSiren: one branch emitting [v, t0, quantiles],
    registered under the submodule name `net` so that state_dicts from before the
    refactor load unchanged. Equivalent config:

        type: MultiBranchSiren
        branches:
          - {name: net, keys: [v, t0, quantiles]}

    For the split topologies, use MultiBranchSiren directly:
        [{keys: [v, t0]}, {keys: [quantiles]}]          # split representation
        [{keys: [v]}, {keys: [t0]}, {keys: [quantiles]}] # split everything
    """

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
        super().__init__(
            in_features=in_features,
            n_quantile=n_quantile,
            outermost_linear=outermost_linear,
            branches=[
                dict(
                    name="net",
                    keys=["v", "t0", "quantiles"],
                    hidden_features=hidden_features,
                    hidden_layers=hidden_layers,
                    first_omega_0=first_omega_0,
                    hidden_omega_0=hidden_omega_0,
                )
            ],
            **kwargs,
        )
        self.n_quantile = n_quantile
        # preserve the legacy split (see PcaSiren) -- total is unchanged either way
        self.out_features = [2, n_quantile]
