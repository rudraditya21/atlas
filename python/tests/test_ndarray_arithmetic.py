import math

import numpy as np
import pytest

import atlas


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (atlas.add, [[2.0, 3.0], [4.0, 5.0]]),
        (atlas.subtract, [[0.0, 1.0], [2.0, 3.0]]),
        (atlas.multiply, [[1.0, 2.0], [3.0, 4.0]]),
        (atlas.divide, [[1.0, 2.0], [3.0, 4.0]]),
    ],
)
def test_arithmetic_supports_scalars(
    operation: object, expected: list[list[float]]
) -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    scalar = 1.0

    assert operation(values, scalar).tolist() == expected


def test_arithmetic_supports_broadcasting_and_views() -> None:
    lhs = np.arange(6.0).reshape(2, 3).T
    rhs = np.array([10.0, 20.0])

    result = atlas.add(lhs, rhs)

    assert result.flags.c_contiguous
    assert result.tolist() == [[10.0, 23.0], [11.0, 24.0], [12.0, 25.0]]


def test_arithmetic_preserves_empty_dimensions() -> None:
    result = atlas.multiply(np.empty((2, 0, 3)), 2.0)

    assert result.shape == (2, 0, 3)
    assert result.size == 0


def test_float_arithmetic_preserves_nan_and_infinity() -> None:
    result = atlas.divide(np.array([math.nan, math.inf, 1.0]), 0.0)

    assert math.isnan(result[0])
    assert math.isinf(result[1])
    assert math.isinf(result[2])


def test_arithmetic_reports_shape_and_integer_division_errors() -> None:
    with pytest.raises(ValueError, match="invalid broadcast"):
        atlas.add(np.zeros((2, 3)), np.zeros((4, 2)))

    with pytest.raises(ValueError, match="division by zero"):
        atlas.divide(np.array([1], dtype=np.int64), 0)
