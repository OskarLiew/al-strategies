from al_strats.ensemble_methods import (
    UncertaintyRankSum,
    ensemble_classification_entropy,
    kl_max_divergence,
    lowest_maximum_confidence,
    max_min_confidence_margin,
    max_min_confidence_ratio,
)
from al_strats.sampler import ConfidenceSampler
from al_strats.uncertainty_methods import (
    classification_entropy,
    least_confidence,
    margin_of_confidence,
    ratio_of_confidence,
)
from tests.common import CONFIDENCE, CONFIDENCE_3D


def test_uncertainty_sampler_multiple_samples() -> None:
    sampler = ConfidenceSampler(least_confidence)
    sample_idx = sampler(CONFIDENCE, n_samples=3)
    assert sample_idx.tolist() == [2, 1, 0]


def test_uncertainty_sampler_more_than_all_samples() -> None:
    sampler = ConfidenceSampler(least_confidence)
    sample_idx = sampler(CONFIDENCE, n_samples=100)
    assert len(sample_idx) == len(CONFIDENCE)


def test_uncertainty_sampler_least_confidence() -> None:
    sampler = ConfidenceSampler(least_confidence)
    sample_idx = sampler(CONFIDENCE)
    assert sample_idx.tolist() == [2]


def test_uncertainty_sampler_margin_of_confidence() -> None:
    sampler = ConfidenceSampler(margin_of_confidence)
    sample_idx = sampler(CONFIDENCE)
    assert sample_idx.tolist() == [2]


def test_uncertainty_sampler_ratio_of_confidence() -> None:
    sampler = ConfidenceSampler(ratio_of_confidence)
    sample_idx = sampler(CONFIDENCE)
    assert sample_idx.tolist() == [2]


def test_uncertainty_sampler_entropy() -> None:
    sampler = ConfidenceSampler(classification_entropy)
    sample_idx = sampler(CONFIDENCE)
    assert sample_idx.tolist() == [2]


def test_uncertainty_sampler_rank_order() -> None:
    urc = UncertaintyRankSum(least_confidence)
    sampler = ConfidenceSampler(urc)
    sample_idx = sampler(CONFIDENCE_3D)
    assert sample_idx.tolist() == [1]


def test_uncertainty_sampler_lowest_max_confidence() -> None:
    sampler = ConfidenceSampler(lowest_maximum_confidence)
    sample_idx = sampler(CONFIDENCE_3D)
    assert sample_idx.tolist() == [2]


def test_uncertainty_sampler_max_min_confidence_margin() -> None:
    sampler = ConfidenceSampler(max_min_confidence_margin)
    sample_idx = sampler(CONFIDENCE_3D)
    assert sample_idx.tolist() == [2]


def test_uncertainty_sampler_max_min_confidence_ratioy() -> None:
    sampler = ConfidenceSampler(max_min_confidence_ratio)
    sample_idx = sampler(CONFIDENCE_3D)
    assert sample_idx.tolist() == [2]


def test_uncertainty_sampler_ensemble_entropy() -> None:
    sampler = ConfidenceSampler(ensemble_classification_entropy)
    sample_idx = sampler(CONFIDENCE_3D)
    assert sample_idx.tolist() == [1]


def test_uncertainty_sampler_kl_max_div() -> None:
    sampler = ConfidenceSampler(kl_max_divergence)
    sample_idx = sampler(CONFIDENCE_3D)
    assert sample_idx.tolist() == [2]
