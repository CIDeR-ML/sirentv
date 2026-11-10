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


def test_masked_bin_bias_mean_over_masked_bins_only():
    from sirentv.eval.utils import masked_bin_bias

    pred = torch.tensor([[[3.0, 1.0, 100.0]]])
    target = torch.tensor([[[1.0, 1.0, 1.0]]])
    mask = torch.tensor([[[True, True, False]]])
    # bins: 2|3-1|/4 = 1, 0 -> mean 0.5; the unmasked outlier is ignored
    assert masked_bin_bias(pred, target, mask).item() == pytest.approx(0.5)


def test_quantile_spacing_centred_and_one_sided():
    from sirentv.eval.utils import quantile_spacing

    t = torch.tensor([0.0, 1.0, 3.0, 6.0])
    assert quantile_spacing(t).tolist() == [1.0, 1.5, 2.5, 3.0]


def test_vis_stats_mean_ignores_below_threshold():
    from sirentv.eval.utils import VisStatsAccumulator

    acc = VisStatsAccumulator(n_pmts=2, threshold=0.5, device="cpu")
    acc.update(torch.tensor([[1.0, 0.1], [3.0, 2.0]]))
    assert acc.finalize()["mean"].tolist() == [2.0, 2.0]
