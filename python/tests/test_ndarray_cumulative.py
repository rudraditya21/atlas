import numpy as np
import pytest

import atlas


@pytest.mark.parametrize(
    ("operation", "values", "expected"),
    [
        (atlas.cumsum, np.array([1, 2, 3], dtype=np.int32), [1, 3, 6]),
        (atlas.cumprod, np.array([1, 2, 3], dtype=np.int32), [1, 2, 6]),
        (atlas.cumsum, np.array([1.5, 2.0, -0.5]), [1.5, 3.5, 3.0]),
        (atlas.cumprod, np.array([1.5, 2.0, -0.5]), [1.5, 3.0, -1.5]),
    ],
)
def test_cumulative_operations_preserve_numeric_dtypes(
    operation: object, values: np.ndarray, expected: list[int | float]
) -> None:
    result = operation(values)

    assert result.dtype == values.dtype
    assert result.tolist() == expected


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (atlas.cumsum, [[1, 5], [7, 12], [15, 21]]),
        (atlas.cumprod, [[1, 4], [8, 40], [120, 720]]),
    ],
)
def test_cumulative_operations_follow_logical_order_for_transposed_views(
    operation: object, expected: list[list[int]]
) -> None:
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64).T

    result = operation(values)

    assert not values.flags.c_contiguous
    assert result.shape == values.shape
    assert result.tolist() == expected


@pytest.mark.parametrize("operation", [atlas.cumsum, atlas.cumprod])
def test_cumulative_operations_preserve_empty_arrays(operation: object) -> None:
    values = np.empty((2, 0, 3), dtype=np.float64)

    result = operation(values)

    assert result.shape == values.shape
    assert result.dtype == values.dtype
    assert result.size == 0
