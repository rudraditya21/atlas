import numpy as np
import pytest

import atlas


def test_concatenate_supports_multiple_axes() -> None:
    lhs = np.array([[1, 2], [3, 4]], dtype=np.int32)
    rhs = np.array([[5, 6], [7, 8]], dtype=np.int32)

    assert atlas.concatenate([lhs, rhs], 0).tolist() == [[1, 2], [3, 4], [5, 6], [7, 8]]
    assert atlas.concatenate([lhs, rhs], 1).tolist() == [[1, 2, 5, 6], [3, 4, 7, 8]]


def test_concatenate_defaults_to_axis_zero_and_accepts_tuples() -> None:
    arrays = (np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32))

    np.testing.assert_array_equal(atlas.concatenate(arrays), np.concatenate(arrays))


def test_concatenate_supports_non_contiguous_views() -> None:
    lhs = np.arange(6, dtype=np.float64).reshape(2, 3).T
    rhs = np.arange(6, 12, dtype=np.float64).reshape(2, 3).T

    result = atlas.concatenate([lhs, rhs], 1)

    assert not lhs.flags.c_contiguous
    assert not rhs.flags.c_contiguous
    np.testing.assert_array_equal(result, np.concatenate([lhs, rhs], axis=1))


def test_concatenate_rejects_empty_inputs() -> None:
    with pytest.raises(atlas.ShapeError, match="at least one array is required"):
        atlas.concatenate([], 0)


def test_concatenate_rejects_mixed_dtypes_and_incompatible_shapes() -> None:
    with pytest.raises(TypeError, match="unsupported NumPy dtype"):
        atlas.concatenate(
            [np.array([1], dtype=np.int32), np.array([1.0], dtype=np.float32)], 0
        )

    with pytest.raises(
        atlas.ShapeError, match="non-concatenation dimensions must match"
    ):
        atlas.concatenate([np.ones((2, 2)), np.ones((3, 3))], 0)
