import numpy as np
import pytest

import atlas


def test_unique_returns_sorted_integer_and_float_values() -> None:
    integers = np.array([3, 1, 3, 2], dtype=np.int32)
    floats = np.array([np.nan, 2.0, np.nan, 1.0], dtype=np.float64)

    integer_result = atlas.unique(integers)
    float_result = atlas.unique(floats)

    np.testing.assert_array_equal(integer_result, [1, 2, 3])
    np.testing.assert_array_equal(float_result, [1.0, 2.0, np.nan])
    assert integer_result.dtype == np.dtype(np.int32)
    assert float_result.dtype == np.dtype(np.float64)


def test_unique_uses_logical_values_from_views() -> None:
    value = np.array([[3, 1, 2], [1, 3, 2]], dtype=np.int16).T

    assert not value.flags.c_contiguous
    np.testing.assert_array_equal(atlas.unique(value), np.unique(value))


def test_unique_accepts_an_explicit_flattened_axis() -> None:
    value = np.array([[3, 1], [2, 1]], dtype=np.int32)

    np.testing.assert_array_equal(atlas.unique(value, axis=None), np.unique(value))


def test_unique_returns_requested_index_outputs() -> None:
    value = np.array([[3, 1, 3], [2, 1, 2]], dtype=np.int32)

    result = atlas.unique(
        value, return_index=True, return_inverse=True, return_counts=True
    )
    expected = np.unique(
        value, return_index=True, return_inverse=True, return_counts=True
    )

    assert isinstance(result, tuple)
    for actual, expected_value in zip(result, expected, strict=True):
        np.testing.assert_array_equal(actual, expected_value)


def test_unique_returns_selected_outputs_in_numpy_order() -> None:
    value = np.array([2, 1, 2, 3], dtype=np.int32)

    values, counts = atlas.unique(value, return_counts=True)
    expected_values, expected_counts = np.unique(value, return_counts=True)

    np.testing.assert_array_equal(values, expected_values)
    np.testing.assert_array_equal(counts, expected_counts)


def test_unique_rejects_non_flattened_axes() -> None:
    with pytest.raises(ValueError, match="axis=None"):
        atlas.unique(np.array([[1, 2]]), axis=0)


def test_unique_handles_empty_inputs() -> None:
    result = atlas.unique(np.empty((2, 0), dtype=np.uint8))

    assert result.shape == (0,)
    assert result.dtype == np.dtype(np.uint8)
