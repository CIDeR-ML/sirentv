from __future__ import annotations

from sirentv.models import MODELS
from sirentv.models.multibranch import MultiBranchSiren


@MODELS.register_module()
class PcaSiren(MultiBranchSiren):
    """Single SIREN trunk outputting PCA coefficients + visibility + log_t0.

    Thin wrapper over MultiBranchSiren: one branch emitting [v, t0, coeffs], registered
    under the submodule name `net` so that state_dicts from before the refactor load
    unchanged. Equivalent config:

        type: MultiBranchSiren
        branches:
          - {name: net, keys: [v, t0, coeffs]}
    """

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
        super().__init__(
            in_features=in_features,
            n_components=n_components,
            outermost_linear=outermost_linear,
            branches=[
                dict(
                    name="net",
                    keys=["v", "t0", "coeffs"],
                    hidden_features=hidden_features,
                    hidden_layers=hidden_layers,
                    first_omega_0=first_omega_0,
                    hidden_omega_0=hidden_omega_0,
                )
            ],
            **kwargs,
        )
        # preserve the legacy split so output_scale init lists written against
        # [2, n_components] keep validating identically (the total is unchanged either way)
        self.out_features = [2, n_components]
