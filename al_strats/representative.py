from typing import Callable, Union

import numpy as np
import numpy.typing as npt
from sklearn.metrics import pairwise_kernels

SimilarityMetric = Union[str, Callable[..., float]]
RepresentativenessStrategy = Callable[..., npt.NDArray]


class RepresentativeSampler:
    def __init__(self, adaptive: bool = False) -> None:
        """The representative sampler helps you sample using representative sampling
        methods.

        Sampling can either be one-shot, or adaptive during the same active-learning
        cycle, by sampling one item at a time, and then in each iteration sample as
        if that item had been removed from the unlabeled items and placed in the
        training set.

        Args:
            adaptive (bool, optional): Use the adaptive strategy. Defaults to False.
        """
        self.adaptive = adaptive

    def __call__(
        self,
        unlabeled_data: npt.ArrayLike,
        train_data: npt.ArrayLike,
        n_samples: int = 1,
        metric: SimilarityMetric = "linear",
    ) -> npt.NDArray:
        """Sample from the unlabeled dataset using representative sampling

        Args:
            unlabeled_data (npt.ArrayLike): Data to sample from as a 2D
                array-like object (n_items, n_features)
            train_data (npt.ArrayLike): Data from the training set as a 2D
                array-like object (n_items, n_features)
            n_samples (int, optional): Number of items to sample. Defaults to 1.
            metric (SimilarityMetric, optional): Metric to use for similarity
                calculations. Defaults to "linear".

        Returns:
            npt.NDArray: 1D array of sample indices (n_samples, )
        """
        if self.adaptive:
            return adative_representative_sampling(
                unlabeled_data, train_data, n_samples, metric
            )

        score = representativeness_score(unlabeled_data, train_data, metric)
        return np.argsort(-score)[:n_samples]


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


def adative_representative_sampling(
    unlabeled_data: npt.ArrayLike,
    train_data: npt.ArrayLike,
    n_samples: int = 1,
    metric: SimilarityMetric = "linear",
) -> npt.NDArray:
    """This sampling strategy is fundamentally the same as the `representativeness_score`,
    except only one item is sampled at a time and then in the next iteration, items are
    sampled as if the sample was moved to the train dataset. This way the strategy can be
    adaptive in the same active learning iteration.

    Since we need to calculate the representativeness anew for every iterations, this
    strategy will be roughly `n_samples` times slower than using the non-adaptive version.

    Args:
        unlabeled_data (npt.ArrayLike): Data to sample from as a 2D
            array-like object (n_items, n_features)
        train_data (npt.ArrayLike): Data from the training set as a 2D
            array-like object (n_items, n_features)
        n_samples (int, optional): Number of items to sample. Defaults to 1.
        metric (SimilarityMetric, optional): Metric to use for similarity
            calculations. Defaults to "linear".

    Returns:
        npt.NDArray: 1D array of sample indices (n_samples, )
    """
    samples = []
    unlabeled_data = np.atleast_2d(unlabeled_data).copy()
    train_data = np.atleast_2d(train_data).copy()
    indices = np.arange(len(unlabeled_data), dtype=int)
    for _ in range(min(n_samples, unlabeled_data.shape[0])):
        score = representativeness_score(unlabeled_data, train_data, metric)
        max_idx = np.argmax(score)
        samples.append(indices[max_idx])

        train_data = np.append(
            train_data, unlabeled_data[max_idx].reshape(1, -1), axis=0
        )
        unlabeled_data = np.delete(unlabeled_data, max_idx, axis=0)
        indices = np.delete(indices, max_idx, axis=0)
    return np.array(samples)
