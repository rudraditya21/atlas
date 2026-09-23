import numpy as np

import atlas


def test_argwhere_supports_boolean_and_numeric_truthiness() -> None:
    boolean = np.array([[True, False], [False, True]])
    numeric = np.array([[0, -2], [3, 0]], dtype=np.int16)

    expected = np.array([[0, 0], [1, 1]], dtype=np.int64)
    np.testing.assert_array_equal(atlas.argwhere(boolean), expected)
    np.testing.assert_array_equal(atlas.argwhere(numeric), np.array([[0, 1], [1, 0]]))


def test_argwhere_uses_logical_coordinates_for_views() -> None:
    value = np.array([[0, 2, 0], [3, 0, 4]], dtype=np.float64).T

    assert not value.flags.c_contiguous
    np.testing.assert_array_equal(
        atlas.argwhere(value), np.array([[0, 1], [1, 0], [2, 1]])
    )


def test_argwhere_returns_int64_coordinates_for_empty_arrays() -> None:
    result = atlas.argwhere(np.empty((2, 0), dtype=np.uint8))

    assert result.shape == (0, 2)
    assert result.dtype == np.dtype(np.int64)
