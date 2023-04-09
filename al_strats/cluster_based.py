from typing import Any, Callable, Protocol, Union
import numpy as np

import numpy.typing as npt
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestCentroid

DistanceMetric = Union[str, Callable[..., float]]
ClusterBasedStrategy = Callable[..., npt.NDArray]


class Clusterer(Protocol):
    def fit(self, X: npt.ArrayLike) -> Any:  # pylint: disable=invalid-name
        ...

    def predict(
        self, X: npt.ArrayLike  # pylint: disable=invalid-name
    ) -> npt.ArrayLike:
        ...


class ClusterBasedSampler:
    def __init__(
        self, sampling_strategy: ClusterBasedStrategy, clusterer: Clusterer
    ) -> None:
        self.strategy = sampling_strategy
        self.clusterer = clusterer

    def __call__(
        self, data: npt.ArrayLike, n_samples: int = 1, **kwargs: Any
    ) -> npt.NDArray:
        self.clusterer.fit(data)
        cluster_labels = self.clusterer.predict(data)
        sample_indices = self.strategy(data, cluster_labels, n_samples, **kwargs)
        return sample_indices[:n_samples]


def sample_cluster_centroids(
    data: npt.ArrayLike,
    cluster_labels: npt.ArrayLike,
    n_samples: int = 1,
    metric: DistanceMetric = "euclidean",
) -> npt.NDArray:
    """Sample items closest to the cluster centroid. These points correspond
    to the most representative items in each cluster

    Args:
        data (npt.ArrayLike): Data to sample from as a 2D array-like object (n_items, n_features)
        cluster_labels (npt.ArrayLike): Cluster membership labels as a 1D
            array-like object. Used to calculate cluster centroids.
        n_samples (int, optional): Number of items to sample. Defaults to 1.
        metric (DistanceMetric, optional): Metric to use for distance calculations.
            Defaults to "euclidean".

    Returns:
        npt.NDArray: 1D array of sample indices
    """
    data = np.atleast_2d(data)
    cluster_labels = np.atleast_1d(cluster_labels)

    centroids = _get_centroids(data, cluster_labels, metric)
    cluster_indices = np.unique(cluster_labels)
    n_cluster_samples = _distribute_number(n_samples, len(cluster_indices))

    indices = np.arange(len(cluster_labels), dtype=int)
    samples = []
    for i_cluster, centroid, n_samples_ in zip(
        cluster_indices, centroids, n_cluster_samples
    ):
        if not n_samples_:
            break
        cluster_members = data[cluster_labels == i_cluster]
        cluster_member_indices = indices[cluster_labels == i_cluster]
        distance_indices = (
            pairwise_distances(cluster_members, centroid.reshape(1, -1))
            .argsort(axis=0)
            .flatten()
        )
        samples.append(cluster_member_indices[distance_indices[:n_samples_]])
    return np.concatenate(samples)


def sample_cluster_outliers(
    data: npt.ArrayLike,
    cluster_labels: npt.ArrayLike,
    n_samples: int = 1,
    metric: DistanceMetric = "euclidean",
) -> npt.NDArray:
    """Sample cluster members farthest from the centroid of the cluster.
    These can represent potentially interesting items that are otherwise
    missed by other sampling strategies

    Args:
        data (npt.ArrayLike): Data to sample from as a 2D array-like
            object (n_items, n_features)
        cluster_labels (npt.ArrayLike): Cluster membership labels as a 1D
            array-like object. Used to calculate cluster centroids.
        n_samples (int, optional): Number of items to sample. Defaults to 1.
        metric (DistanceMetric, optional): Metric to use for distance calculations.
            Defaults to "euclidean".

    Returns:
        npt.NDArray: 1D array of sample indices
    """
    data = np.atleast_2d(data)
    cluster_labels = np.atleast_1d(cluster_labels)
    centroids = _get_centroids(data, cluster_labels, metric)
    cluster_indices = np.unique(cluster_labels)
    indices = np.arange(len(cluster_labels), dtype=int)
    n_cluster_samples = _distribute_number(n_samples, len(cluster_indices))

    samples = []
    for i_cluster, centroid, n_samples_ in zip(
        cluster_indices, centroids, n_cluster_samples
    ):
        if not n_samples_:
            break
        cluster_members = data[cluster_labels == i_cluster]
        cluster_member_indices = indices[cluster_labels == i_cluster]
        distance_indices = (
            pairwise_distances(cluster_members, centroid.reshape(1, -1))
            .argsort(axis=0)
            .flatten()
        )
        samples.append(cluster_member_indices[distance_indices[-n_samples_:]])
    return np.concatenate(samples)


def sample_random_cluster_members(
    _: Any,
    cluster_labels: npt.ArrayLike,
    n_samples: int = 1,
    seed: int = None,
) -> npt.NDArray:
    """Sample items randomly from each cluster.

    Args:
        _ (Any): Only included for interface consistency
        cluster_labels (npt.ArrayLike): Cluster membership labels as a 1D
            array-like object. Used to calculate cluster centroids.
        n_samples (int, optional): Number of items to sample. Defaults to 1.
        seed (int, optional): The seed of the random number generator.
            Defaults to None.

    Returns:
        npt.NDArray: 1D array of sample indices
    """
    rng = np.random.default_rng(seed)
    cluster_labels = np.atleast_1d(cluster_labels)
    indices = np.arange(len(cluster_labels), dtype=int)

    samples = []
    cluster_indices = np.unique(cluster_labels)
    n_cluster_samples = _distribute_number(n_samples, len(cluster_indices))
    for i_cluster, n_samples_ in zip(cluster_indices, n_cluster_samples):
        cluster_indices = indices[cluster_labels == i_cluster]
        cluster_samples = rng.choice(cluster_indices, n_samples_, replace=False)
        samples.append(cluster_samples)
    return np.concatenate(samples)


def _get_centroids(
    data: npt.NDArray, cluster_labels: npt.NDArray, metric: DistanceMetric = "euclidean"
) -> npt.NDArray:
    nearest_centroid = NearestCentroid(metric=metric).fit(data, cluster_labels)
    return nearest_centroid.centroids_


def _distribute_number(to_distribute: int, arr_len: int) -> npt.NDArray:
    """Evenly distributes integers into an array of size arr_len"""
    arr = np.ones(arr_len, dtype=int) * (to_distribute // arr_len)
    arr[: to_distribute % arr_len] += 1
    return arr
