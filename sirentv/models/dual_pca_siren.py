from __future__ import annotations

from typing import List

from sirentv.models import MODELS
from sirentv.models.multibranch import MultiBranchSiren, _as_list


@MODELS.register_module()
class DualPcaSiren(MultiBranchSiren):
    """Two independent SIREN branches: one for vis + log_t0, one for PCA coefficients.

    Thin wrapper over MultiBranchSiren, registering the branches under the submodule
    names `vt0_net` and `coeff_net` so that state_dicts from before the refactor load
    unchanged -- which is what makes the existing coeffs_only / high_omega checkpoints
    reusable in Stage 1. Equivalent config:

        type: MultiBranchSiren
        branches:
          - {name: vt0_net,   keys: [v, t0],   hidden_features: 256, first_omega_0: 30}
          - {name: coeff_net, keys: [coeffs],  hidden_features: 512, first_omega_0: 150}

    The [vt0_branch, coeff_branch] list convention is preserved for every argument: a
    bare scalar applies uniformly to both branches (backward compatible with every
    existing config), a 2-element list lets them differ. Per-branch omega_0 exists
    specifically to test whether coeff_net's steep-near-PMT / flat-far-field difficulty
    is a SIREN bandwidth limit -- raising omega_0 raises the range of spatial frequencies
    the branch can represent, at the cost of needing a cleaner signal (dynamic_weight, or
    a longer warmup) to keep that extra bandwidth from showing up as ringing in the
    otherwise smooth far field.
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
        # legacy value: [vt0 width, coeff width]; identical to what this topology
        # produces anyway, kept explicit so the intent is obvious
        self.out_features = [2, n_components]
