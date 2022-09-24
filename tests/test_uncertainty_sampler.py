from al_strats.sampler import UncertaintySampler
from al_strats.uncertainty_metrics import (
    least_confidence,
    margin_of_confidence,
    ratio_of_confidence,
    classification_entropy,
)
from al_strats.ensemble_metrics import (
    UncertaintyRankSum,
    lowest_maximum_confidence,
    max_min_confidence_margin,
    max_min_confidence_ratio,
    ensemble_classification_entropy,
    kl_max_divergence,
)
from tests.common import CONFIDENCE, CONFIDENCE_3D


def test_uncertainty_sampler_multiple_samples() -> None:
    sampler = UncertaintySampler(least_confidence)
    sample_idx = sampler(CONFIDENCE, n_samples=3)
    assert sample_idx.tolist() == [2, 1, 0]


def test_uncertainty_sampler_least_confidence() -> None:
    sampler = UncertaintySampler(least_confidence)
    sample_idx = sampler(CONFIDENCE)
    assert sample_idx.tolist() == [2]


def test_uncertainty_sampler_margin_of_confidence() -> None:
    sampler = UncertaintySampler(margin_of_confidence)
    sample_idx = sampler(CONFIDENCE)
    assert sample_idx.tolist() == [2]


def test_uncertainty_sampler_ratio_of_confidence() -> None:
    sampler = UncertaintySampler(ratio_of_confidence)
    sample_idx = sampler(CONFIDENCE)
    assert sample_idx.tolist() == [2]


def test_uncertainty_sampler_entropy() -> None:
    sampler = UncertaintySampler(classification_entropy)
    sample_idx = sampler(CONFIDENCE)
    assert sample_idx.tolist() == [2]


def test_uncertainty_sampler_rank_order() -> None:
    urc = UncertaintyRankSum(least_confidence)
    sampler = UncertaintySampler(urc)
    sample_idx = sampler(CONFIDENCE_3D)
    assert sample_idx.tolist() == [1]


def test_uncertainty_sampler_lowest_max_confidence() -> None:
    sampler = UncertaintySampler(lowest_maximum_confidence)
    sample_idx = sampler(CONFIDENCE_3D)
    assert sample_idx.tolist() == [2]


def test_uncertainty_sampler_max_min_confidence_margin() -> None:
    sampler = UncertaintySampler(max_min_confidence_margin)
    sample_idx = sampler(CONFIDENCE_3D)
    assert sample_idx.tolist() == [2]


def test_uncertainty_sampler_max_min_confidence_ratioy() -> None:
    sampler = UncertaintySampler(max_min_confidence_ratio)
    sample_idx = sampler(CONFIDENCE_3D)
    assert sample_idx.tolist() == [2]


def test_uncertainty_sampler_ensemble_entropy() -> None:
    sampler = UncertaintySampler(ensemble_classification_entropy)
    sample_idx = sampler(CONFIDENCE_3D)
    assert sample_idx.tolist() == [1]


def test_uncertainty_sampler_kl_max_div() -> None:
    sampler = UncertaintySampler(kl_max_divergence)
    sample_idx = sampler(CONFIDENCE_3D)
    assert sample_idx.tolist() == [2]
