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
    with pytest.raises(atlas.ShapeError, match="invalid broadcast"):
        atlas.add(np.zeros((2, 3)), np.zeros((4, 2)))

    with pytest.raises(atlas.NumericError, match="division by zero"):
        atlas.divide(np.array([1], dtype=np.int64), 0)


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        ("int8", [-2, 3]),
        ("uint8", [2, 3]),
        ("int32", [-2, 3]),
        ("uint64", [2, 3]),
    ],
)
def test_integer_dtypes_preserve_construction_arithmetic_and_comparison_output(
    dtype: str, values: list[int]
) -> None:
    source = np.array(values, dtype=dtype)

    constructed = atlas.ones([2], dtype=dtype)
    added = atlas.add(source, 1)
    comparison = atlas.greater(source, 0)

    assert constructed.dtype == np.dtype(dtype)
    assert added.dtype == np.dtype(dtype)
    assert added.tolist() == (source + 1).tolist()
    assert comparison.dtype == np.dtype(bool)
    assert comparison.tolist() == (source > 0).tolist()


def test_array_operations_reject_mixed_dtypes() -> None:
    with pytest.raises(TypeError, match="unsupported NumPy dtype"):
        atlas.add(np.array([1], dtype=np.int32), np.array([1.0], dtype=np.float32))

    with pytest.raises(TypeError, match="unsupported NumPy dtype"):
        atlas.equal(np.array([1], dtype=np.uint8), np.array([1], dtype=np.int16))


@pytest.mark.parametrize(
    ("values", "scalar", "expected"),
    [
        (np.array([1, 2], dtype=np.int8), 3, [4, 5]),
        (np.array([1, 2], dtype=np.uint8), 3, [4, 5]),
    ],
)
def test_array_operations_accept_scalars_that_fit_the_left_dtype(
    values: np.ndarray, scalar: int, expected: list[int]
) -> None:
    result = atlas.add(values, scalar)

    assert result.dtype == values.dtype
    assert result.tolist() == expected


@pytest.mark.parametrize(
    ("values", "scalar"),
    [
        (np.array([1], dtype=np.int8), 128),
        (np.array([1], dtype=np.uint8), -1),
    ],
)
def test_array_operations_reject_scalars_outside_the_left_dtype(
    values: np.ndarray, scalar: int
) -> None:
    with pytest.raises(OverflowError):
        atlas.add(values, scalar)
