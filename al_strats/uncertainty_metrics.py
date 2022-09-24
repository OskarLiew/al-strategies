import numpy.typing as npt
import numpy as np


def least_confidence(confidence: npt.ArrayLike) -> npt.NDArray:
    confidence = np.atleast_2d(confidence)
    return 1 - np.max(confidence, axis=1)


def margin_of_confidence(confidence: npt.ArrayLike) -> npt.NDArray:
    confidence = np.atleast_2d(confidence)
    sorted_confidence = np.sort(confidence, axis=1)  # asc
    return 1 - (sorted_confidence[:, -1] - sorted_confidence[:, -2])


def ratio_of_confidence(confidence: npt.ArrayLike) -> npt.NDArray:
    confidence = np.atleast_2d(confidence)
    sorted_confidence = np.sort(confidence, axis=1)
    return sorted_confidence[:, -2] / sorted_confidence[:, -1]


def classification_entropy(confidence: npt.ArrayLike) -> npt.NDArray:
    confidence = np.atleast_2d(confidence)
    logs = np.log2(confidence)
    return -np.sum(confidence * logs, axis=1)
