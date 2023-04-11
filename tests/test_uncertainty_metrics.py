from al_strats.uncertainty_methods import (
    classification_entropy,
    least_confidence,
    margin_of_confidence,
    ratio_of_confidence,
)
from tests.common import CONFIDENCE, approx


def test_least_confidence() -> None:
    uncertainty = least_confidence(CONFIDENCE)
    assert approx(uncertainty, [0.3, 0.5, 0.66])


def test_margin_of_confidence() -> None:
    uncertainty = margin_of_confidence(CONFIDENCE)
    assert approx(uncertainty, [0.5, 0.9, 0.99])


def test_ratio_of_confidence() -> None:
    uncertainty = ratio_of_confidence(CONFIDENCE)
    assert approx(uncertainty, [0.286, 0.8, 0.971], abs=1e-3)


def test_classification_entropy() -> None:
    uncertainty = classification_entropy(CONFIDENCE)
    assert approx(uncertainty, [1.157, 1.36, 1.585], abs=1e-3)
