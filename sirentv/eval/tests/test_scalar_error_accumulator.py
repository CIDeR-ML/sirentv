import pytest
import torch

from sirentv.eval.utils import ScalarErrorAccumulator


def test_symmetric_signed_errors_do_not_cancel():
    """Regression test: ScalarErrorAccumulator previously accumulated the signed
    error (pred - target) instead of |pred - target|, so symmetric over/under
    predictions silently canceled to a near-zero mean even though every single
    prediction was off by the same magnitude. If that regresses, this mean
    collapses to ~0 instead of 2.0.
    """
    pred = torch.tensor([2.0, -2.0, 2.0, -2.0])
    target = torch.zeros(4)
    mask = torch.ones(4, dtype=torch.bool)

    acc = ScalarErrorAccumulator(device="cpu")
    acc.update(pred, target, mask)
    result = acc.finalize()

    assert result["count"].item() == 4
    assert result["mean"].item() == pytest.approx(2.0)
    assert result["std"].item() == pytest.approx(0.0, abs=1e-6)


def test_respects_mask():
    pred = torch.tensor([5.0, 100.0])
    target = torch.tensor([0.0, 0.0])
    mask = torch.tensor([True, False])

    acc = ScalarErrorAccumulator(device="cpu")
    acc.update(pred, target, mask)
    result = acc.finalize()

    assert result["count"].item() == 1
    assert result["mean"].item() == pytest.approx(5.0)
