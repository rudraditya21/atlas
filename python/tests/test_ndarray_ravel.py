import numpy as np

import atlas


def test_ravel_flattens_contiguous_arrays_in_logical_order() -> None:
    value = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

    result = atlas.ravel(value)

    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5, 6], dtype=np.int32))
    assert result.dtype == np.dtype(np.int32)


def test_ravel_flattens_transposed_views_in_logical_order() -> None:
    value = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64).T

    assert not value.flags.c_contiguous
    np.testing.assert_array_equal(atlas.ravel(value), np.array([1, 4, 2, 5, 3, 6]))


def test_ravel_normalizes_scalar_arrays() -> None:
    result = atlas.ravel(np.array(7, dtype=np.int16))

    np.testing.assert_array_equal(result, np.array([7], dtype=np.int16))
    assert result.dtype == np.dtype(np.int16)


def test_ravel_preserves_empty_arrays() -> None:
    result = atlas.ravel(np.empty((2, 0, 3), dtype=np.uint8))

    assert result.shape == (0,)
    assert result.dtype == np.dtype(np.uint8)
