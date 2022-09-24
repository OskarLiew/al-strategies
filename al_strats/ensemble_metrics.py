from typing import Callable
import numpy.typing as npt
import numpy as np


class UncertaintyRankSum:
    def __init__(
        self,
        uncertainty_metric: Callable[[npt.ArrayLike], npt.NDArray],
    ) -> None:
        self.uncertainty_metric = uncertainty_metric

    def __call__(self, confidences: npt.ArrayLike) -> npt.NDArray:
        confidences = np.atleast_3d(confidences)
        uncertainties = _calculate_confidences_modelwise(
            confidences, self.uncertainty_metric
        )
        return _rank_sum_criterion(uncertainties)


def _calculate_confidences_modelwise(
    confidences: npt.NDArray, uncertainty_metric: Callable[[npt.ArrayLike], npt.NDArray]
):
    uncertainties = np.zeros((confidences.shape[0], confidences.shape[2]))
    for i in range(confidences.shape[2]):
        uncertainties[:, i] = uncertainty_metric(confidences[:, :, i])
    return uncertainties


def _rank_sum_criterion(uncertainties: npt.ArrayLike) -> npt.NDArray:
    uncertainties = np.atleast_2d(uncertainties)
    ranks = np.zeros((uncertainties.shape[0], uncertainties.shape[1]))
    for i in range(uncertainties.shape[1]):
        ranks[:, i] = _rank_elements(uncertainties[:, i])
    return np.sum(ranks, axis=1)


def _rank_elements(array: npt.NDArray) -> npt.NDArray:
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks


def lowest_maximum_confidence(confidences: npt.ArrayLike) -> npt.NDArray:
    confidences = np.atleast_3d(confidences)
    max_confidences = np.max(confidences, axis=1)  # Most confident prediction
    return np.min(max_confidences, axis=1)  # Minimum of model confidences


def max_min_confidence_margin(confidences: npt.ArrayLike) -> npt.NDArray:
    confidences = np.atleast_3d(confidences)
    max_confidences = np.max(confidences, axis=1)  # Top prediction confidence
    sorted_confidence = np.sort(max_confidences, axis=1)  # Sort model confidences, asc
    return sorted_confidence[:, -1] - sorted_confidence[:, 0]  # max - min


def max_min_confidence_ratio(confidences: npt.ArrayLike) -> npt.NDArray:
    confidences = np.atleast_3d(confidences)
    max_confidences = np.max(confidences, axis=1)  # Top prediction confidence
    sorted_confidence = np.sort(max_confidences, axis=1)  # Sort model confidences, asc
    return 1 - sorted_confidence[:, 0] / sorted_confidence[:, -1]  # min / max
