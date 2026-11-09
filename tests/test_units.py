"""Small checks of shared helpers; no LUT or GPU needed."""
import os

import numpy as np
import pytest
import torch


def test_select_voxel_indices_subsample_and_x_min():
    from sirentv.data.common import select_voxel_indices

    pos = np.zeros((10, 3))
    pos[:, 0] = np.arange(10) * 10.0 - 50.0  # x = -50 .. 40
    idx = select_voxel_indices({"subsample_factor": 2, "x_min": -25}, 10, pos, rank=1, tag="t")
    assert idx.tolist() == [4, 6, 8]
    assert select_voxel_indices({"max_len": 3}, 10, pos, rank=1, tag="t").tolist() == [0, 1, 2]


def test_resolve_config():
    from sirentv.training.utils import CONFIG_DIR, resolve_config

    assert resolve_config("abl.yaml", archive=True) == os.path.join(CONFIG_DIR, "archive", "abl.yaml")
    assert resolve_config("train_q3_t0align_s3df.yaml") == os.path.join(CONFIG_DIR, "train_q3_t0align_s3df.yaml")
    assert resolve_config(__file__) == __file__


def test_clip_grad_dict_requires_every_parameter_matched():
    from sirentv.training.utils import _clip_grad_

    net = torch.nn.ModuleDict({"v_net": torch.nn.Linear(2, 1), "t0_net": torch.nn.Linear(2, 1)})
    for p in net.parameters():
        p.grad = torch.full_like(p, 10.0)
    _clip_grad_({"v_net": 1.0, "t0_net": 1.0}, net)
    for key in ("v_net", "t0_net"):
        norm = torch.norm(torch.stack([p.grad.norm() for p in net[key].parameters()]))
        assert norm.item() == pytest.approx(1.0, rel=1e-5)
    with pytest.raises(ValueError):
        _clip_grad_({"v_net": 1.0}, net)


def test_multibranch_output_keys_and_shapes():
    from sirentv.models.multibranch import MultiBranchSiren

    net = MultiBranchSiren(
        in_features=6, n_quantile=8, hidden_features=16, hidden_layers=1,
        branches=[{"name": "vt", "keys": ["v", "t0"], "init_output_bias": {"v": 0.5}},
                  {"name": "q", "keys": ["quantiles"]}],
    )
    out = net(torch.zeros(4, 3, 6))
    assert out["v"].shape == (4, 3) and out["t0"].shape == (4, 3) and out["quantiles"].shape == (4, 3, 8)
    assert net.vt.net[-1].bias[0].item() == pytest.approx(0.5)
    with pytest.raises(ValueError):
        MultiBranchSiren(branches=[{"keys": ["v"]}, {"keys": ["v"]}])


def test_csv_logger_writes_every_log_step(tmp_path):
    from sirentv.utils.log import CSVLogger

    logger = CSVLogger({"logger": {"run_dir": str(tmp_path), "dir_name": "logs", "log_every_nsteps": 2}})
    for it in range(1, 5):
        logger.record(["iter", "loss"], [it, 1.5e-7 * it])
        logger.step(it)
    logger.close()
    lines = open(logger.logfile).read().splitlines()
    assert lines == ["iter,loss", "2.0000e+00,3.0000e-07", "4.0000e+00,6.0000e-07"]
