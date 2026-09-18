import numpy as np
import pytest

import atlas


def test_asarray_copies_logical_values_and_preserves_dtype() -> None:
    source = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).T
    result = atlas.asarray(source)

    assert result.dtype == np.dtype(np.float32)
    assert result.flags.c_contiguous
    assert result.tolist() == [[1.0, 3.0], [2.0, 4.0]]


@pytest.mark.parametrize(
    ("constructor", "shape", "dtype", "expected"),
    [
        (atlas.zeros, [], "float64", 0.0),
        (atlas.ones, [2, 0, 3], "int64", [[], []]),
        (atlas.zeros, [2, 2], "float32", [[0.0, 0.0], [0.0, 0.0]]),
    ],
)
def test_zeros_and_ones_support_shapes_and_dtypes(
    constructor: object, shape: list[int], dtype: str, expected: object
) -> None:
    result = constructor(shape, dtype=dtype)

    assert result.dtype == np.dtype(dtype)
    assert result.tolist() == expected


def test_full_supports_boolean_dtype() -> None:
    result = atlas.full([2, 2], True, dtype="bool")

    assert result.dtype == np.dtype(bool)
    assert result.tolist() == [[True, True], [True, True]]


def test_arange_supports_default_and_selected_dtypes() -> None:
    assert atlas.arange(4).tolist() == [0.0, 1.0, 2.0, 3.0]
    assert atlas.arange(1, 5, 2, dtype="int64").tolist() == [1, 3]


@pytest.mark.parametrize("shape", [[2**64], [2, 2**64]])
def test_constructors_reject_invalid_dimensions(shape: list[int]) -> None:
    with pytest.raises((OverflowError, ValueError)):
        atlas.zeros(shape)
