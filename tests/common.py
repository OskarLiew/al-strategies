from typing import Any, List
import pytest
import numpy.typing as npt

CONFIDENCE = [[0.7, 0.1, 0.2], [0.5, 0.4, 0.1], [0.33, 0.33, 0.34]]
CONFIDENCE_3D = [  # Shape: (n_items, classes, n_models)
    [[0.7, 0.8], [0.1, 0.1], [0.2, 0.1]],
    [[0.5, 0.2], [0.4, 0.1], [0.1, 0.7]],
    [[0.33, 0.9], [0.33, 0.09], [0.34, 0.01]],
]


def approx(uncertainty: npt.NDArray, expected: List[Any], **kwargs: Any) -> bool:
    return uncertainty.tolist() == pytest.approx(expected, **kwargs)
