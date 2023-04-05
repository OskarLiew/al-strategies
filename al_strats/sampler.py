from typing import Callable, Optional
import numpy.typing as npt
import numpy as np


class ConfidenceSampler:
    def __init__(
        self, sampling_strategy: Callable[[npt.ArrayLike], npt.NDArray]
    ) -> None:
        self.uncertainty_metric = sampling_strategy

    def __call__(self, confidence: npt.ArrayLike, n_samples: int = 1) -> npt.NDArray:
        uncertainty = self.uncertainty_metric(confidence)
        return np.argsort(-uncertainty)[:n_samples]


class RandomSampler:
    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, dataset_size: int, n_samples: int = 1) -> npt.NDArray:
        n_samples = min(dataset_size, n_samples)
        sample_ids = self.rng.choice(dataset_size, size=n_samples, replace=False)
        return sample_ids
