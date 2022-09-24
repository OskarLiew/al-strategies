from typing import Callable
import numpy.typing as npt
import numpy as np


class UncertaintySampler:
    def __init__(
        self, uncertainty_metric: Callable[[npt.ArrayLike], npt.NDArray]
    ) -> None:
        self.uncertainty_metric = uncertainty_metric

    def __call__(self, confidence: npt.ArrayLike, n_samples: int = 1) -> npt.NDArray:
        uncertainty = self.uncertainty_metric(confidence)
        return np.argsort(-uncertainty)[:n_samples]
