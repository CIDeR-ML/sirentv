"""
Generic multi-branch SIREN.

One independent `slar.base.Siren` per branch, each branch emitting one or more named
outputs. The representation key is just another output name, so every topology works
identically for PCA coefficients and for raw quantiles:

    topology                     PCA representation                quantile representation
    -------------------------------------------------------------------------------------
    single trunk                 [{keys: [v, t0, coeffs]}]         [{keys: [v, t0, quantiles]}]
                                 (== legacy PcaSiren)              (== legacy QuantileSiren)
    split representation         [{keys: [v, t0]}, {keys: [coeffs]}]
                                 (== legacy DualPcaSiren)          [{keys: [v, t0]}, {keys: [quantiles]}]
    split everything (triple)    [{keys: [v]}, {keys: [t0]}, {keys: [coeffs]}]
                                                                   [{keys: [v]}, {keys: [t0]}, {keys: [quantiles]}]

Nothing in this class is specific to either representation -- `coeffs` and `quantiles`
differ only in their output width (n_components vs n_quantile), so any topology above can
be built for either by swapping that one key.

Why branches at all: the point of splitting is that each branch gets its own
`first_omega_0` / `hidden_omega_0` / depth / width, i.e. its own representable spatial
bandwidth. A shared trunk forces one bandwidth compromise across outputs whose spectra
differ (which is what the kx/ky/kz power-spectrum diagnostic measures). Splitting buys
nothing except through those per-branch knobs.

Note what is deliberately NOT a branch: `*_grad_frob`. The predicted gradient magnitude
is not a network output -- SirenTV.forward autograd-differentiates a branch's value
output with respect to position (see compute_grad_frob_hutchinson).

Branch naming and checkpoint compatibility
------------------------------------------
Each branch is registered as a submodule under its own `name`, so the resulting
state_dict keys are chosen by config rather than by position. The legacy wrapper classes
pass the names the hand-written modules used (`net`, `vt0_net`, `coeff_net`), which is
what lets checkpoints trained before this refactor load unchanged -- a ModuleList would
have produced `nets.0.*` and silently broken every existing checkpoint.
"""

from __future__ import annotations

from typing import List

from slar.base import Siren
from torch import nn

from sirentv.models import MODELS


def _as_list(value, n):
    """Broadcast a scalar to a list of length n; pass a list through, padding by repeat.

    Keeps the legacy wrappers backward compatible: every existing config may give either
    a bare scalar (applies to all branches) or a per-branch list.
    """
    if isinstance(value, (int, float)):
        return [value] * n
    value = list(value)
    if len(value) < n:
        value = value + [value[-1]] * (n - len(value))
    return value[:n]


@MODELS.register_module()
class MultiBranchSiren(nn.Module):
    """Multi-branch SIREN driven by a branch spec.

    branches: list of dicts, each with
        keys             : [str]     output names this branch emits, in slice order
        name             : str       submodule name (default "branch{i}")
        out_widths       : {k: int}  override a key's output width
        hidden_features  : int       defaults to the top-level value
        hidden_layers    : int       defaults to the top-level value
        first_omega_0    : float     defaults to the top-level value
        hidden_omega_0   : float     defaults to the top-level value
        outermost_linear : bool      defaults to the top-level value

    The top-level hidden_features/hidden_layers/first_omega_0/hidden_omega_0 act as
    defaults for branches that do not set their own. Output widths default to 1 for
    `v`/`t0`, `n_components` for `coeffs` and `n_quantile` for `quantiles`; both
    representation keys are handled the same way, so a topology written for one works for
    the other by swapping that key.
    """

    def __init__(
        self,
        in_features: int = 6,
        branches: List[dict] | None = None,
        n_components: int = 50,
        n_quantile: int = 256,
        outermost_linear: bool = True,
        hidden_features: int = 256,
        hidden_layers: int = 6,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        **kwargs,
    ):
        super().__init__()
        if not branches:
            raise ValueError(
                "MultiBranchSiren needs a non-empty `branches` list, e.g. "
                "[{'keys': ['v','t0']}, {'keys': ['quantiles']}]"
            )

        self.n_components = n_components
        self.n_quantile = n_quantile
        default_widths = {
            "v": 1,
            "t0": 1,
            "coeffs": n_components,
            "quantiles": n_quantile,
        }

        self._branch_names: List[str] = []
        self._branch_keys: List[List[str]] = []
        self._branch_widths: List[List[int]] = []
        seen = set()

        for i, spec in enumerate(branches):
            keys = list(spec["keys"])
            if not keys:
                raise ValueError(f"branch {i} has an empty `keys` list")
            # A key emitted by two branches would make the output dict order-dependent and
            # silently drop one of them, so reject it rather than pick a winner.
            dup = seen.intersection(keys)
            if dup:
                raise ValueError(
                    f"branch {i} repeats output key(s) already emitted: {sorted(dup)}"
                )
            seen.update(keys)

            overrides = spec.get("out_widths", {}) or {}
            widths = []
            for k in keys:
                if k in overrides:
                    widths.append(int(overrides[k]))
                elif k in default_widths:
                    widths.append(int(default_widths[k]))
                else:
                    raise ValueError(
                        f"branch {i}: unknown output key {k!r} with no width. Give it "
                        f"out_widths: {{{k}: <int>}}, or use one of "
                        f"{sorted(default_widths)}."
                    )

            name = spec.get("name", f"branch{i}")
            net = Siren(
                in_features,
                spec.get("hidden_features", hidden_features),
                spec.get("hidden_layers", hidden_layers),
                sum(widths),
                spec.get("outermost_linear", outermost_linear),
                spec.get("first_omega_0", first_omega_0),
                spec.get("hidden_omega_0", hidden_omega_0),
            )
            # add_module rather than a ModuleList: the submodule name (and therefore every
            # state_dict key) comes from the spec, which is what keeps the legacy wrappers
            # checkpoint-compatible.
            self.add_module(name, net)
            self._branch_names.append(name)
            self._branch_keys.append(keys)
            self._branch_widths.append(widths)

        # Consumed only for the output_scale length assertion and n_outs (see
        # SirenTV.out_features), both of which care about the total, not the split.
        self.out_features = [sum(w) for w in self._branch_widths]

    @property
    def branch_topology(self):
        """[(name, keys)] -- handy for logging which topology a run actually built."""
        return list(zip(self._branch_names, self._branch_keys))

    def forward(self, x, *args, **kwargs):
        out = {}
        for name, keys, widths in zip(
            self._branch_names, self._branch_keys, self._branch_widths
        ):
            y = getattr(self, name)(x)
            offset = 0
            for key, width in zip(keys, widths):
                chunk = y[..., offset:offset + width]
                # scalar outputs are (B, N_pmt), matching what the losses and the dataset
                # targets expect -- not (B, N_pmt, 1)
                out[key] = chunk.squeeze(-1) if width == 1 else chunk
                offset += width
        return out
