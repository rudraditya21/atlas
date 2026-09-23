import numpy as np
import pytest

import atlas


def test_roll_supports_positive_negative_and_axis_shifts() -> None:
    value = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

    np.testing.assert_array_equal(atlas.roll(value, 2), np.roll(value, 2))
    np.testing.assert_array_equal(atlas.roll(value, -1), np.roll(value, -1))
    np.testing.assert_array_equal(
        atlas.roll(value, 1, axis=0), np.roll(value, 1, axis=0)
    )
    np.testing.assert_array_equal(
        atlas.roll(value, -1, axis=-1), np.roll(value, -1, axis=-1)
    )


def test_roll_uses_logical_values_from_views() -> None:
    value = np.arange(6, dtype=np.float64).reshape(2, 3).T

    assert not value.flags.c_contiguous
    np.testing.assert_array_equal(
        atlas.roll(value, 1, axis=1), np.roll(value, 1, axis=1)
    )


def test_roll_supports_scalar_and_sequence_axis_pairs() -> None:
    value = np.arange(24, dtype=np.int32).reshape(2, 3, 4)

    np.testing.assert_array_equal(
        atlas.roll(value, [1, -2], axis=[0, 2]), np.roll(value, [1, -2], axis=[0, 2])
    )
    np.testing.assert_array_equal(
        atlas.roll(value, 1, axis=[0, 2]), np.roll(value, 1, axis=[0, 2])
    )


def test_roll_rejects_incompatible_shift_axis_lengths() -> None:
    with pytest.raises(atlas.NumericError, match="matching lengths"):
        atlas.roll(np.ones((2, 3)), [1, 2], axis=[0, 1, 0])


def test_roll_preserves_empty_arrays() -> None:
    value = np.empty((2, 0), dtype=np.uint8)

    result = atlas.roll(value, 3, axis=1)

    assert result.shape == (2, 0)
    assert result.dtype == np.dtype(np.uint8)


def test_roll_translates_invalid_axes() -> None:
    with pytest.raises(atlas.AxisError, match="axis 2 is out of bounds"):
        atlas.roll(np.ones((2, 2)), 1, axis=2)
