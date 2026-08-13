from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from sirentv.eval import evaluate
from sirentv.models import SirenTV
from sirentv.train import train
from sirentv.training.utils import step_scheduler


def test_checkpoint_roundtrip_is_standalone(tiny_cfg: dict, tmp_path: Path) -> None:
    original_cfg = copy.deepcopy(tiny_cfg)
    model = SirenTV(tiny_cfg)
    position = torch.zeros(2, 3)
    expected = model(position)
    checkpoint = tmp_path / "model.ckpt"

    model.save_state(checkpoint, epoch=1.5, iteration=7)
    loaded = SirenTV.load(str(checkpoint))
    actual = loaded(position)

    assert tiny_cfg == original_cfg
    assert loaded._checkpoint_epoch == 1.5
    assert loaded._checkpoint_iteration == 7
    assert torch.equal(actual["v"], expected["v"])
    assert torch.equal(actual["t"], expected["t"])


def test_legacy_checkpoint_has_actionable_error(tiny_cfg: dict) -> None:
    model_dict = SirenTV(tiny_cfg).model_dict()
    model_dict.pop("data_cfg")

    with pytest.raises(ValueError, match="Pass a full configuration dictionary"):
        SirenTV.create_from_model_dict(model_dict)


def test_scheduler_step_contracts() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    step = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=0
    )

    optimizer.step()
    step_scheduler(step, 10.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    step_scheduler(plateau, 2.0)
    step_scheduler(plateau, 3.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_one_iteration_training(tiny_cfg: dict) -> None:
    cfg = copy.deepcopy(tiny_cfg)
    cfg["train"]["max_iterations"] = 1
    cfg["data"]["loader"]["batch_size"] = 8

    train(cfg)

    log_files = list(Path(cfg["logger"]["dir_name"]).glob("version-*/log.csv"))
    assert len(log_files) == 1
    assert len(log_files[0].read_text().splitlines()) >= 2


def test_evaluation_writes_complete_results(tiny_cfg: dict, tmp_path: Path) -> None:
    cfg = copy.deepcopy(tiny_cfg)
    cfg["data"]["loader"]["batch_size"] = 8
    output = tmp_path / "evaluation.pt"

    evaluate(cfg, str(output))
    results = torch.load(output, weights_only=True)

    assert results["meta"]["n_positions"] == 8
    assert results["meta"]["n_pmts"] == 2
    assert results["visibility_all"]["pred"].shape == (8, 2)
    assert results["time_bias"]["mean"].shape == (8,)
