import numpy as np

import atlas


def test_flatten_returns_an_independent_row_major_output_for_views() -> None:
    value = np.arange(6, dtype=np.int32).reshape(2, 3).T

    result = atlas.flatten(value)

    assert not value.flags.c_contiguous
    np.testing.assert_array_equal(result, [0, 3, 1, 4, 2, 5])
    assert result.dtype == np.dtype(np.int32)
    assert not np.shares_memory(result, value)
    result[0] = 99
    assert value[0, 0] == 0


def test_flatten_and_ravel_have_logical_order_for_views() -> None:
    value = np.arange(6, dtype=np.float64).reshape(2, 3).T

    np.testing.assert_array_equal(atlas.flatten(value), atlas.ravel(value))


def test_flatten_normalizes_empty_arrays_to_one_dimension() -> None:
    result = atlas.flatten(np.empty((2, 0, 3), dtype=np.uint8))

    assert result.shape == (0,)
    assert result.dtype == np.dtype(np.uint8)
