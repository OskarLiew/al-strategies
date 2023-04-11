from typing import Callable, Union

import numpy as np
import numpy.typing as npt
from sklearn.metrics import pairwise_kernels

SimilarityMetric = Union[str, Callable[..., float]]
RepresentativenessStrategy = Callable[..., npt.NDArray]


def representativeness_score(
    unlabeled_data: npt.ArrayLike,
    train_data: npt.ArrayLike,
    metric: SimilarityMetric = "linear",
) -> npt.NDArray:
    """Measures representativeness of unlabeled data points relative to the
    training set. Models the gap between training data and the domain where
    the model will be deployed.

    Args:
        unlabeled_data (npt.ArrayLike): Data to sample from as a 2D
            array-like object (n_items, n_features)
        train_data (npt.ArrayLike): Data from the training set as a 2D
            array-like object (n_items, n_features)
        metric (SimilarityMetric, optional): Metric to use for similarity
            calculations. Defaults to "linear".

    Returns:
        npt.NDArray: 1D array of representativeness values. Higher is more
            representative (n_items, )
    """
    unlabeled_data = np.atleast_2d(unlabeled_data)
    train_data = np.atleast_2d(train_data)

    unlabeled_centroid = unlabeled_data.mean(axis=0)
    train_centroid = train_data.mean(axis=0)

    dists_unlabeled = pairwise_kernels(
        unlabeled_data, unlabeled_centroid.reshape(1, -1), metric=metric
    ).flatten()
    dists_train = pairwise_kernels(
        unlabeled_data, train_centroid.reshape(1, -1), metric=metric
    ).flatten()
    return dists_unlabeled - dists_train
