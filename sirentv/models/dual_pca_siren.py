from __future__ import annotations

from typing import List

from sirentv.models import MODELS
from sirentv.models.multibranch import MultiBranchSiren, _as_list


@MODELS.register_module()
class DualPcaSiren(MultiBranchSiren):
    """Branches `vt0_net` -> [v, t0] and `coeff_net` -> [coeffs] (legacy submodule names).

    Each argument is a scalar for both branches or a [vt0, coeff] list.
    """

    def __init__(
        self,
        in_features: int = 6,
        hidden_features: List[int] | int = [256, 512],
        hidden_layers: List[int] | int = [3, 3],
        n_components: int = 50,
        outermost_linear: bool = True,
        first_omega_0: List[float] | float = 30.0,
        hidden_omega_0: List[float] | float = 30.0,
        **kwargs,
    ):
        hf = _as_list(hidden_features, 2)
        hl = _as_list(hidden_layers, 2)
        fo = _as_list(first_omega_0, 2)
        ho = _as_list(hidden_omega_0, 2)

        super().__init__(
            in_features=in_features,
            n_components=n_components,
            outermost_linear=outermost_linear,
            branches=[
                dict(
                    name="vt0_net",
                    keys=["v", "t0"],
                    hidden_features=hf[0],
                    hidden_layers=hl[0],
                    first_omega_0=fo[0],
                    hidden_omega_0=ho[0],
                ),
                dict(
                    name="coeff_net",
                    keys=["coeffs"],
                    hidden_features=hf[1],
                    hidden_layers=hl[1],
                    first_omega_0=fo[1],
                    hidden_omega_0=ho[1],
                ),
            ],
            **kwargs,
        )
        self.out_features = [2, n_components]
