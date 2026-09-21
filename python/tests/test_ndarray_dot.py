import numpy as np
import pytest

import atlas


def test_dot_supports_vector_vector_products() -> None:
    assert (
        atlas.dot(
            np.array([1, 2, 3], dtype=np.int32), np.array([4, 5, 6], dtype=np.int32)
        )
        == 32
    )


def test_dot_supports_matrix_vector_products() -> None:
    matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    vector = np.array([1, 2, 3], dtype=np.float64)

    assert atlas.dot(matrix, vector).tolist() == [14.0, 32.0]


def test_dot_supports_matrix_matrix_products() -> None:
    lhs = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    rhs = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.int64)

    assert atlas.dot(lhs, rhs).tolist() == [[58, 64], [139, 154]]


def test_dot_supports_transposed_views() -> None:
    lhs = np.arange(6, dtype=np.float32).reshape(3, 2).T
    rhs = np.arange(6, 12, dtype=np.float32).reshape(2, 3).T

    assert not lhs.flags.c_contiguous
    assert not rhs.flags.c_contiguous
    np.testing.assert_array_equal(atlas.dot(lhs, rhs), np.dot(lhs, rhs))


def test_dot_translates_incompatible_shapes_to_shape_error() -> None:
    with pytest.raises(atlas.ShapeError, match="shape mismatch for dot"):
        atlas.dot(np.array([1, 2]), np.array([1, 2, 3]))
