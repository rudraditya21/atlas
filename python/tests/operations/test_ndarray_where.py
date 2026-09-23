import numpy as np
import pytest

import atlas


def test_where_supports_array_and_scalar_operands() -> None:
    condition = np.array([True, False, True])
    values = np.array([1, 2, 3], dtype=np.int32)

    assert atlas.where(condition, values, 0).tolist() == [1, 0, 3]
    assert atlas.where(condition, -1, values).tolist() == [-1, 2, -1]


def test_where_with_only_a_condition_returns_per_axis_indices() -> None:
    condition = np.array([[False, True], [True, False]])

    result = atlas.where(condition)

    assert isinstance(result, tuple)
    assert [indices.tolist() for indices in result] == [[0, 1], [1, 0]]


def test_where_rejects_only_one_selection_operand() -> None:
    condition = np.array([True, False])

    with pytest.raises(TypeError, match="both x and y"):
        atlas.where(condition, 1)
    with pytest.raises(TypeError, match="both x and y"):
        atlas.where(condition, y=0)


def test_where_supports_broadcasting_and_views() -> None:
    condition = np.array([[True], [False]])
    x = np.array([[1, 2]], dtype=np.float64)
    y = np.array([[0, 3], [0, 4]], dtype=np.float64).T

    result = atlas.where(condition, x, y)

    assert not y.flags.c_contiguous
    assert result.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_where_rejects_mixed_array_dtypes() -> None:
    with pytest.raises(TypeError, match="unsupported NumPy dtype"):
        atlas.where(
            np.array([True]),
            np.array([1], dtype=np.int32),
            np.array([1.0], dtype=np.float32),
        )


def test_where_rejects_non_boolean_conditions() -> None:
    with pytest.raises(TypeError, match="unsupported NumPy dtype"):
        atlas.where(np.array([1], dtype=np.int32), np.array([1]), np.array([2]))


def test_where_translates_incompatible_broadcasts_to_shape_error() -> None:
    with pytest.raises(atlas.ShapeError, match="invalid broadcast"):
        atlas.where(np.ones((2, 3), dtype=bool), np.ones((4, 2)), np.ones((4, 2)))
