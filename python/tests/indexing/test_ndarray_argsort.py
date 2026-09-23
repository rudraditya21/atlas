import numpy as np

import atlas


def test_argsort_handles_ties_nans_and_index_dtype() -> None:
    values = np.array([2.0, np.nan, 1.0, 1.0])

    result = atlas.argsort(values)

    np.testing.assert_array_equal(result, np.array([2, 3, 0, 1], dtype=np.int64))
    assert result.dtype == np.dtype(np.int64)


def test_argsort_supports_axes_and_views() -> None:
    value = np.array([[3, 1, 2], [6, 4, 5]], dtype=np.int32).T

    assert not value.flags.c_contiguous
    np.testing.assert_array_equal(
        atlas.argsort(value, axis=1), np.argsort(value, axis=1)
    )
    np.testing.assert_array_equal(
        atlas.argsort(value, axis=0), np.argsort(value, axis=0)
    )


def test_argsort_without_an_axis_flattens_logical_values() -> None:
    value = np.array([[3, 1, 2], [6, 4, 5]], dtype=np.int32).T

    np.testing.assert_array_equal(
        atlas.argsort(value, axis=None), np.argsort(value, axis=None)
    )


def test_argsort_handles_empty_inputs() -> None:
    result = atlas.argsort(np.empty((2, 0, 3), dtype=np.uint8), axis=1)

    assert result.shape == (2, 0, 3)
    assert result.dtype == np.dtype(np.int64)
