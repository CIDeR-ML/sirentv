from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from examples.create_tiny_photonlib import create_tiny_photonlib


@pytest.fixture
def tiny_photonlib(tmp_path: Path) -> Path:
    return create_tiny_photonlib(tmp_path / "tiny_photonlib.h5")


@pytest.fixture
def tiny_cfg(tiny_photonlib: Path, tmp_path: Path) -> dict:
    config_path = Path(__file__).parents[1] / "config" / "example_tiny.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    cfg["photonlib"]["filepath"] = str(tiny_photonlib)
    cfg["logger"]["dir_name"] = str(tmp_path / "logs")
    return copy.deepcopy(cfg)
