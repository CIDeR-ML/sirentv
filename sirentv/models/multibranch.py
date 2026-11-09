"""Generic multi-branch SIREN: one independent `slar.base.Siren` per branch, each emitting named outputs.

Topologies, for either representation (`coeffs` or `quantiles`, which differ only in width):

    single trunk    [{keys: [v, t0, coeffs]}]                      == PcaSiren / QuantileSiren
    split repr.     [{keys: [v, t0]}, {keys: [coeffs]}]            == DualPcaSiren
    split all       [{keys: [v]}, {keys: [t0]}, {keys: [coeffs]}]

Branches exist so each output gets its own omega_0 / depth / width, i.e. its own spatial
bandwidth. `*_grad_frob` is not a branch: SirenTV.forward differentiates an output w.r.t. position.
Each branch is a submodule under its config `name`, so state_dict keys come from the config;
the wrappers use the legacy names (`net`, `vt0_net`, `coeff_net`) to keep old checkpoints loading.
"""

from __future__ import annotations

from typing import List

import torch
from slar.base import Siren
from torch import nn

from sirentv.models import MODELS


def _as_list(value, n):
    """Broadcast a scalar to length n; pad a shorter list by repeating its last entry."""
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
        init_output_bias : {k: float} or float, starting bias of the final layer per key
        hidden_features / hidden_layers / first_omega_0 / hidden_omega_0 / outermost_linear:
                           per-branch overrides of the top-level values

    Output widths default to 1 for `v`/`t0`, `n_components` for `coeffs`, `n_quantile` for `quantiles`.
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
            # Why init_output_bias exists: the Poisson loss clamps inv_xform(pred) at eps, and
            # clamp has zero gradient below it. A SIREN starts near 0, which inv_xform maps
            # onto/below eps, so v can start dead and never train (3 of 4 seeds measured).
            # A bias near the target mean starts it in the live region for any seed.
            self._init_output_bias(net, name, spec.get("init_output_bias"), keys, widths)
            self.add_module(name, net)
            self._branch_names.append(name)
            self._branch_keys.append(keys)
            self._branch_widths.append(widths)

        # only the total matters (output_scale length check, n_outs)
        self.out_features = [sum(w) for w in self._branch_widths]

    @staticmethod
    def _init_output_bias(net, branch_name, bias_spec, keys, widths):
        """Set the final layer's bias per output key ({key: value}, or a scalar for every key)."""
        if bias_spec is None:
            return
        if isinstance(bias_spec, (int, float)):
            bias_spec = {k: float(bias_spec) for k in keys}
        unknown = sorted(set(bias_spec) - set(keys))
        if unknown:
            raise ValueError(
                f"branch {branch_name!r}: init_output_bias names key(s) {unknown} that this "
                f"branch does not emit (it emits {keys})"
            )

        # outermost_linear=True gives a bare nn.Linear, otherwise a SineLayer wrapping one
        final = net.net[-1]
        linear = final if isinstance(final, nn.Linear) else getattr(final, "linear", None)
        if linear is None or linear.bias is None:
            raise ValueError(
                f"branch {branch_name!r}: cannot apply init_output_bias -- the final layer "
                f"({type(final).__name__}) exposes no bias to set"
            )

        with torch.no_grad():
            offset = 0
            for key, width in zip(keys, widths):
                if key in bias_spec:
                    linear.bias[offset:offset + width] = float(bias_spec[key])
                offset += width

    @property
    def branch_topology(self):
        """[(name, keys)] of the built branches."""
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
                # scalar outputs are (B, N_pmt), like the targets
                out[key] = chunk.squeeze(-1) if width == 1 else chunk
                offset += width
        return out
