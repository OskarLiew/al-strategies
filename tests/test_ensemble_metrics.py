import numpy as np

from al_strats.ensemble_metrics import (
    UncertaintyRankSum,
    _rank_elements,
    lowest_maximum_confidence,
    max_min_confidence_margin,
    max_min_confidence_ratio,
    ensemble_classification_entropy,
    kl_max_divergence,
)
from al_strats.uncertainty_metrics import least_confidence
from tests.common import approx, CONFIDENCE_3D


def test_uncertainty_rank_order() -> None:
    urc = UncertaintyRankSum(least_confidence)
    ranks = urc(CONFIDENCE_3D)
    assert approx(ranks, [1, 3, 2])


def test_rank_elements() -> None:
    ranks = _rank_elements(np.array([0, 10, 3, 2, 7]))
    assert ranks.tolist() == [0, 4, 2, 1, 3]


def test_lowest_maximum_confidence() -> None:
    lmc = lowest_maximum_confidence(CONFIDENCE_3D)
    assert approx(lmc, [0.3, 0.5, 0.66])


def test_max_min_confidence_margin() -> None:
    mmcm = max_min_confidence_margin(CONFIDENCE_3D)
    assert approx(mmcm, [0.1, 0.2, 0.56])


def test_max_min_confidence_ratio() -> None:
    mmcr = max_min_confidence_ratio(CONFIDENCE_3D)
    assert approx(mmcr, [0.125, 0.286, 0.622], abs=1e-3)


def test_ensemble_classification_entropy() -> None:
    mmcr = ensemble_classification_entropy(CONFIDENCE_3D)
    assert approx(mmcr, [2.079, 2.518, 2.101], abs=1e-3)


def test_kl_max_divergence() -> None:
    kl_max_div = kl_max_divergence(CONFIDENCE_3D)
    assert approx(kl_max_div, [0.011, 0.228, 0.238], abs=1e-3)
