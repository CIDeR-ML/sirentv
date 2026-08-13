from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from sirentv.data.builder import create_dataloader
from sirentv.models import SirenTV
from sirentv.models.branched import BranchedSiren
from sirentv.models.parallel import ParallelSiren


@pytest.mark.parametrize("lazy", [False, True])
def test_photonlib_dataset_contract(tiny_cfg: dict, lazy: bool) -> None:
    cfg = copy.deepcopy(tiny_cfg)
    cfg["photonlib"]["lazy"] = lazy

    batch = next(iter(create_dataloader(cfg)))

    assert batch["position"].shape == (2, 3)
    assert batch["target"]["v"].shape == (2, 2)
    assert batch["target"]["t"].shape == (2, 2, 8)
    assert batch["meta"]["v_linear"].shape == (2, 2)
    assert batch["meta"]["t_linear"].shape == (2, 2, 8)


@pytest.mark.parametrize(
    ("model_type", "hidden_features", "hidden_layers", "expected_keys"),
    [
        ("BranchedSiren", [4, 4, 4], [2, 2, 2], {"v", "t", "t0"}),
        ("ParallelSiren", [4, 4], [2, 2], {"v", "t", "t0"}),
        ("PcaSiren", 4, 2, {"v", "coeffs", "t0"}),
        ("DualPcaSiren", [4, 4], [2, 2], {"v", "coeffs", "t0"}),
    ],
)
def test_supported_models_forward_and_backward(
    tiny_cfg: dict,
    model_type: str,
    hidden_features,
    hidden_layers,
    expected_keys: set[str],
) -> None:
    cfg = copy.deepcopy(tiny_cfg)
    network = {
        "type": model_type,
        "in_features": 3,
        "hidden_features": hidden_features,
        "hidden_layers": hidden_layers,
        "outermost_linear": True,
    }
    if "Pca" in model_type:
        network["n_components"] = 3
    else:
        network["out_features"] = [1, 9]
    cfg["model"]["network"] = network

    model = SirenTV(cfg)
    output = model(torch.zeros(2, 3))

    assert expected_keys <= output.keys()
    assert output["v"].shape == (2, 2)
    if "t" in output:
        assert output["t"].shape == (2, 2, 8)
    else:
        assert output["coeffs"].shape == (2, 2, 3)

    sum(value.sum() for key, value in output.items() if key != "correct_mask").backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_gradient_supervision_reaches_model_parameters(tiny_cfg: dict) -> None:
    cfg = copy.deepcopy(tiny_cfg)
    cfg["model"]["network"]["hidden_features"] = [4, 4, 4]

    model = SirenTV(cfg)
    output = model(torch.zeros(2, 3), return_gradients=True)
    loss = output["grad_mags_transformed"].sum()
    loss.backward()

    assert output["grad_mags_transformed"].shape == (2, 2)
    gradients = [
        parameter.grad
        for parameter in model.model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert sum(gradient.abs().sum().item() for gradient in gradients) > 0


class _Identity(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _ConstantWaveform(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        shape = (*value.shape[:-1], 4)
        output = torch.zeros(shape, device=value.device)
        output[..., 0] = 0.25
        output[..., 1] = 0.75
        return output


class _ConstantVisibility(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*value.shape[:-1], 1), device=value.device)


def test_branched_t0_uses_dedicated_output_channel() -> None:
    model = BranchedSiren(
        in_features=3,
        hidden_features=[4, 4, 4],
        hidden_layers=[2, 2, 2],
        out_features=[1, 4],
    )
    model.encoder = _Identity()
    model.waveform_decoder = _ConstantWaveform()
    model.vis_decoder = _ConstantVisibility()

    output = model(torch.zeros(1, 2, 3), tau=1.0)

    assert torch.allclose(output["t0"], torch.full((1, 2), 0.75))


def test_parallel_t0_uses_dedicated_output_channel() -> None:
    model = ParallelSiren(
        in_features=3,
        hidden_features=[4, 4],
        hidden_layers=[2, 2],
        out_features=[1, 4],
    )
    model.t0_cdf_net = _ConstantWaveform()
    model.v_net = _ConstantVisibility()

    output = model(torch.zeros(1, 2, 3), tau=1.0)

    expected = torch.sigmoid(torch.tensor(0.25)) * 3
    assert torch.allclose(output["t0"], torch.full((1, 2), expected))
