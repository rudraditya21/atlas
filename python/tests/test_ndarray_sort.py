import numpy as np

import atlas


def test_sort_orders_integer_and_float_lanes() -> None:
    integers = np.array([[3, 1, 2], [6, 4, 5]], dtype=np.int32)
    floats = np.array([3.0, np.nan, 1.0, 2.0], dtype=np.float64)

    np.testing.assert_array_equal(atlas.sort(integers), np.sort(integers))
    np.testing.assert_array_equal(
        atlas.sort(integers, axis=0), np.sort(integers, axis=0)
    )
    np.testing.assert_array_equal(atlas.sort(floats), np.sort(floats))


def test_sort_uses_logical_values_from_views() -> None:
    value = np.array([[3, 1, 2], [6, 4, 5]], dtype=np.float64).T

    assert not value.flags.c_contiguous
    np.testing.assert_array_equal(atlas.sort(value, axis=1), np.sort(value, axis=1))


def test_sort_handles_empty_lanes() -> None:
    value = np.empty((2, 0, 3), dtype=np.uint8)

    result = atlas.sort(value, axis=1)

    assert result.shape == (2, 0, 3)
    assert result.dtype == np.dtype(np.uint8)
