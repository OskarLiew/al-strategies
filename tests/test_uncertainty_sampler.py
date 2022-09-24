from al_strats.sampler import UncertaintySampler
from al_strats.uncertainty_metrics import (
    least_confidence,
    margin_of_confidence,
    ratio_of_confidence,
    classification_entropy,
)
from tests.common import CONFIDENCE


def test_uncertainty_sampler_lc() -> None:
    sampler = UncertaintySampler(least_confidence)
    sample_idx = sampler(CONFIDENCE)
    assert sample_idx.tolist() == [2]


def test_uncertainty_sampler_mc() -> None:
    sampler = UncertaintySampler(margin_of_confidence)
    sample_idx = sampler(CONFIDENCE)
    assert sample_idx.tolist() == [2]


def test_uncertainty_sampler_rc() -> None:
    sampler = UncertaintySampler(ratio_of_confidence)
    sample_idx = sampler(CONFIDENCE)
    assert sample_idx.tolist() == [2]


def test_uncertainty_sampler_ce() -> None:
    sampler = UncertaintySampler(classification_entropy)
    sample_idx = sampler(CONFIDENCE)
    assert sample_idx.tolist() == [2]


def test_uncertainty_sampler_multiple_samples() -> None:
    sampler = UncertaintySampler(least_confidence)
    sample_idx = sampler(CONFIDENCE, n_samples=3)
    assert sample_idx.tolist() == [2, 1, 0]
