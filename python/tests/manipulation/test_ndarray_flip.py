import numpy as np

import atlas


def test_flip_reverses_all_axes_or_a_selected_axis() -> None:
    value = np.array([[1, 2], [3, 4]], dtype=np.int32)

    np.testing.assert_array_equal(atlas.flip(value), np.flip(value))
    np.testing.assert_array_equal(atlas.flip(value, axis=0), np.flip(value, axis=0))
    np.testing.assert_array_equal(atlas.flip(value, axis=-1), np.flip(value, axis=-1))


def test_flip_uses_logical_values_from_views() -> None:
    value = np.arange(6, dtype=np.float64).reshape(2, 3).T

    assert not value.flags.c_contiguous
    np.testing.assert_array_equal(atlas.flip(value, axis=1), np.flip(value, axis=1))


def test_flip_preserves_empty_dimensions() -> None:
    value = np.empty((2, 0, 3), dtype=np.uint8)

    result = atlas.flip(value, axis=-1)

    assert result.shape == (2, 0, 3)
    assert result.dtype == np.dtype(np.uint8)
