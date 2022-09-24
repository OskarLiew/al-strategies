from typing import Any, List
import pytest
import numpy.typing as npt

CONFIDENCE = [[0.7, 0.1, 0.2], [0.5, 0.4, 0.1], [0.33, 0.33, 0.34]]


def approx(uncertainty: npt.NDArray, expected: List[Any], **kwargs: Any) -> bool:
    return uncertainty.tolist() == pytest.approx(expected, **kwargs)
