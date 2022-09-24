from typing import Callable
import numpy.typing as npt
import numpy as np


class UncertaintyRankSum:
    def __init__(
        self,
        uncertainty_metric: Callable[[npt.ArrayLike], npt.NDArray],
    ) -> None:
        """Calculates the uncertainty rank sum using a given uncertainty metric function

        Args:
            uncertainty_metric (Callable[[npt.ArrayLike], npt.NDArray]): Callable that
                takes an array-like object with two dimensions (n_items, n_classes)
                and returns a single uncertainty score for each item in an array.
        """
        self.uncertainty_metric = uncertainty_metric

    def __call__(self, confidences: npt.ArrayLike) -> npt.NDArray:
        """Calculates uncertainty rank sum

        Args:
            confidences (npt.ArrayLike): Prediction confidence with as a 3D
                array-like object (n_items, n_classes, n_models)

        Returns:
            npt.NDArray: Array with shape (n_items, )
        """
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
    """Lowest maximum confidence across all models

    Args:
        confidences (npt.ArrayLike): Prediction confidence with as a 3D
                array-like object (n_items, n_classes, n_models)

    Returns:
        npt.NDArray: Array with shape (n_items, )
    """
    confidences = np.atleast_3d(confidences)
    max_confidences = np.max(confidences, axis=1)  # Most confident prediction
    return 1 - np.min(max_confidences, axis=1)  # Minimum of model confidences


def max_min_confidence_margin(confidences: npt.ArrayLike) -> npt.NDArray:
    """Difference between minimum and maximum confidence across models

    Args:
        confidences (npt.ArrayLike): Prediction confidence with as a 3D
                array-like object (n_items, n_classes, n_models)

    Returns:
        npt.NDArray: Array with shape (n_items, )
    """
    confidences = np.atleast_3d(confidences)
    max_confidences = np.max(confidences, axis=1)  # Top prediction confidence
    sorted_confidence = np.sort(max_confidences, axis=1)  # Sort model confidences, asc
    return sorted_confidence[:, -1] - sorted_confidence[:, 0]  # max - min


def max_min_confidence_ratio(confidences: npt.ArrayLike) -> npt.NDArray:
    """Ratio between minimum and maximum confidence across models

    Args:
        confidences (npt.ArrayLike): Prediction confidence with as a 3D
                array-like object (n_items, n_classes, n_models)

    Returns:
        npt.NDArray: Array with shape (n_items, )
    """
    confidences = np.atleast_3d(confidences)
    max_confidences = np.max(confidences, axis=1)  # Top prediction confidence
    sorted_confidence = np.sort(max_confidences, axis=1)  # Sort model confidences, asc
    return 1 - sorted_confidence[:, 0] / sorted_confidence[:, -1]  # min / max


def ensemble_classification_entropy(confidences: npt.ArrayLike) -> npt.NDArray:
    """Classification entropy across all confidences in all models

    Args:
        confidences (npt.ArrayLike): Prediction confidence with as a 3D
                array-like object (n_items, n_classes, n_models)

    Returns:
        npt.NDArray: Array with shape (n_items, )
    """
    confidences = np.atleast_3d(confidences)
    logs = np.log2(confidences)
    return -(confidences * logs).sum(axis=1).sum(axis=1)


def kl_max_divergence(confidences: npt.ArrayLike) -> npt.NDArray:
    """Calculated KL divergence between model distributions and the
    average distribution, then returns the maximum kl-div over the models.

    Args:
        confidences (npt.ArrayLike): Prediction confidence with as a 3D
                array-like object (n_items, n_classes, n_models)

    Returns:
        npt.NDArray: Array with shape (n_items, )
    """
    confidences = np.atleast_3d(confidences)
    kl_divs = _kl_divergence(confidences)
    return np.max(kl_divs, axis=1)


def _kl_divergence(confidences: npt.NDArray) -> npt.NDArray:
    avg_confidence = np.mean(confidences, axis=2)
    kl_divs = np.zeros((confidences.shape[0], confidences.shape[2]))
    for model_idx in range(confidences.shape[2]):
        model_confidence = confidences[:, :, model_idx]
        log_fraction = np.log(model_confidence / avg_confidence)
        kl_divs[:, model_idx] = np.sum(model_confidence * log_fraction, axis=1)

    return kl_divs
