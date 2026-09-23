import numpy as np
import pytest

import atlas


def test_stack_inserts_dimensions_at_requested_and_negative_axes() -> None:
    lhs = np.array([[1, 2], [3, 4]], dtype=np.int32)
    rhs = np.array([[5, 6], [7, 8]], dtype=np.int32)

    np.testing.assert_array_equal(
        atlas.stack([lhs, rhs], 0), np.stack([lhs, rhs], axis=0)
    )
    np.testing.assert_array_equal(
        atlas.stack([lhs, rhs], -1), np.stack([lhs, rhs], axis=-1)
    )


def test_stack_defaults_to_axis_zero_and_accepts_tuples() -> None:
    arrays = (np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32))

    np.testing.assert_array_equal(atlas.stack(arrays), np.stack(arrays))


def test_stack_supports_non_contiguous_views() -> None:
    lhs = np.arange(6, dtype=np.float64).reshape(2, 3).T
    rhs = np.arange(6, 12, dtype=np.float64).reshape(2, 3).T

    result = atlas.stack([lhs, rhs], 1)

    assert not lhs.flags.c_contiguous
    assert not rhs.flags.c_contiguous
    np.testing.assert_array_equal(result, np.stack([lhs, rhs], axis=1))


def test_stack_supports_scalar_arrays() -> None:
    result = atlas.stack([np.array(1, dtype=np.int64), np.array(2, dtype=np.int64)], 0)

    assert result.tolist() == [1, 2]


def test_stack_validates_input_shapes_and_empty_inputs() -> None:
    with pytest.raises(atlas.ShapeError, match="all input shapes must match"):
        atlas.stack([np.ones((2, 2)), np.ones((2, 3))], 0)

    with pytest.raises(atlas.ShapeError, match="at least one array is required"):
        atlas.stack([], 0)
