import numpy as np
import pytest

import atlas


def test_repeat_supports_counts_and_axes() -> None:
    value = np.array([[1, 2], [3, 4]], dtype=np.int32)

    np.testing.assert_array_equal(atlas.repeat(value, 2), [1, 1, 2, 2, 3, 3, 4, 4])
    np.testing.assert_array_equal(
        atlas.repeat(value, 2, axis=0), [[1, 2], [1, 2], [3, 4], [3, 4]]
    )
    np.testing.assert_array_equal(
        atlas.repeat(value, 2, axis=-1), [[1, 1, 2, 2], [3, 3, 4, 4]]
    )


def test_repeat_uses_logical_values_from_views() -> None:
    value = np.arange(6, dtype=np.float64).reshape(2, 3).T

    assert not value.flags.c_contiguous
    np.testing.assert_array_equal(
        atlas.repeat(value, 2, axis=1), np.repeat(value, 2, axis=1)
    )


def test_repeat_handles_empty_arrays_and_zero_counts() -> None:
    value = np.empty((2, 0), dtype=np.uint8)

    result = atlas.repeat(value, 3, axis=1)

    assert result.shape == (2, 0)
    assert result.dtype == np.dtype(np.uint8)
    assert atlas.repeat(np.array([1, 2]), 0).shape == (0,)


def test_repeat_rejects_negative_counts() -> None:
    with pytest.raises(atlas.NumericError, match="repeats must be nonnegative"):
        atlas.repeat(np.array([1, 2]), -1)
