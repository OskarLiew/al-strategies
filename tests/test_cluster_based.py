from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

from al_strats.cluster_based import (
    ClusterBasedSampler,
    sample_cluster_centroids,
    sample_cluster_outliers,
    sample_random_cluster_members,
)
from tests.common import SEED


def test_cluster_based_sampler() -> None:
    data, target = load_iris(return_X_y=True)
    kmeans = KMeans(3, n_init="auto")
    sampler = ClusterBasedSampler(sample_cluster_centroids, kmeans)
    samples = sampler(data, n_samples=3)
    print(samples)
    assert set(target[samples]) == {0, 1, 2}


def test_cluster_centroids() -> None:
    data, target = load_iris(return_X_y=True)
    samples = sample_cluster_centroids(data, target, 3)
    assert set(target[samples]) == {0, 1, 2}


def test_cluster_centroids_extra_samples() -> None:
    data, target = load_iris(return_X_y=True)
    samples = sample_cluster_centroids(data, target, 5)
    assert set(target[samples]) == {0, 1, 2}
    assert len(samples) == 5


def test_cluster_centroids_fewer_samples() -> None:
    data, target = load_iris(return_X_y=True)
    samples = sample_cluster_centroids(data, target, 2)
    assert set(target[samples]) == {0, 1}
    assert len(samples) == 2


def test_cluster_centroids_more_samples_than_items() -> None:
    data, _ = load_iris(return_X_y=True)
    samples = sample_cluster_centroids(data[:2], [0, 1], 3)
    assert len(samples) == 2


def test_cluster_centroids_other_metric() -> None:
    data, target = load_iris(return_X_y=True)
    samples = sample_cluster_centroids(data, target, 3, metric="manhattan")
    assert set(target[samples]) == {0, 1, 2}


def test_cluster_outliers() -> None:
    data, target = load_iris(return_X_y=True)
    samples = sample_cluster_outliers(data, target, 3)
    assert set(target[samples]) == {0, 1, 2}


def test_cluster_outliers_extra_samples() -> None:
    data, target = load_iris(return_X_y=True)
    samples = sample_cluster_outliers(data, target, 5)
    assert set(target[samples]) == {0, 1, 2}
    assert len(samples) == 5


def test_cluster_outliers_fewer_samples() -> None:
    data, target = load_iris(return_X_y=True)
    samples = sample_cluster_outliers(data, target, 2)
    print(samples)
    print(target[samples])
    assert set(target[samples]) == {0, 1}
    assert len(samples) == 2


def test_cluster_outliers_other_metric() -> None:
    data, target = load_iris(return_X_y=True)
    samples = sample_cluster_outliers(data, target, 3, metric="manhattan")
    assert set(target[samples]) == {0, 1, 2}


def test_cluster_outliers_more_samples_than_items() -> None:
    data, _ = load_iris(return_X_y=True)
    samples = sample_cluster_outliers(data[:2], [0, 1], 3)
    assert len(samples) == 2


def test_cluster_random() -> None:
    data, target = load_iris(return_X_y=True)
    samples = sample_random_cluster_members(data, target, 3, seed=SEED)
    assert set(target[samples]) == {0, 1, 2}


def test_cluster_random_extra_samples() -> None:
    data, target = load_iris(return_X_y=True)
    samples = sample_random_cluster_members(data, target, 5, seed=SEED)
    assert set(target[samples]) == {0, 1, 2}
    assert len(samples) == 5


def test_cluster_random_fewer_samples() -> None:
    data, target = load_iris(return_X_y=True)
    samples = sample_random_cluster_members(data, target, 2, seed=SEED)
    assert set(target[samples]) == {0, 1}
    assert len(samples) == 2


def test_cluster_random_other_metric() -> None:
    data, target = load_iris(return_X_y=True)
    samples = sample_random_cluster_members(data, target, 3, seed=SEED)
    assert set(target[samples]) == {0, 1, 2}


def test_cluster_random_more_samples_than_items() -> None:
    data, _ = load_iris(return_X_y=True)
    samples = sample_random_cluster_members(data[:2], [0, 1], 3)
    assert len(samples) == 2
