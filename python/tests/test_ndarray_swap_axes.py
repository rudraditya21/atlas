import numpy as np
import pytest

import atlas


def test_swap_axes_supports_positive_and_negative_axes() -> None:
    value = np.arange(24, dtype=np.int32).reshape(2, 3, 4)

    np.testing.assert_array_equal(
        atlas.swap_axes(value, 0, 2), np.swapaxes(value, 0, 2)
    )
    np.testing.assert_array_equal(
        atlas.swap_axes(value, -1, -3), np.swapaxes(value, -1, -3)
    )


def test_swap_axes_uses_logical_values_from_views() -> None:
    value = np.arange(24, dtype=np.float64).reshape(2, 3, 4).transpose(2, 0, 1)

    assert not value.flags.c_contiguous
    np.testing.assert_array_equal(
        atlas.swap_axes(value, 0, 1), np.swapaxes(value, 0, 1)
    )


def test_swap_axes_preserves_zero_sized_dimensions() -> None:
    value = np.empty((2, 0, 3), dtype=np.uint8)

    result = atlas.swap_axes(value, 0, 2)

    assert result.shape == (3, 0, 2)
    assert result.dtype == np.dtype(np.uint8)


def test_swap_axes_translates_invalid_axes() -> None:
    with pytest.raises(atlas.AxisError, match="axis 2 is out of bounds"):
        atlas.swap_axes(np.ones((2, 2)), 0, 2)
