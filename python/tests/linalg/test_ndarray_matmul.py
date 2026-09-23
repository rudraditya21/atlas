import numpy as np
import pytest

import atlas


def test_matmul_supports_vector_matrix_products() -> None:
    lhs = np.array([1, 2, 3], dtype=np.int32)
    rhs = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)

    result = atlas.matmul(lhs, rhs)

    assert result.dtype == np.dtype(np.int32)
    assert result.tolist() == [22, 28]


def test_matmul_supports_matrix_vector_products() -> None:
    lhs = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    rhs = np.array([4, 5, 6], dtype=np.float64)

    result = atlas.matmul(lhs, rhs)

    assert result.dtype == np.dtype(np.float64)
    assert result.tolist() == [32.0, 77.0]


def test_matmul_supports_matrix_matrix_products() -> None:
    lhs = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    rhs = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.int64)

    assert atlas.matmul(lhs, rhs).tolist() == [[58, 64], [139, 154]]


def test_matmul_supports_transposed_inputs() -> None:
    lhs = np.arange(6, dtype=np.float32).reshape(3, 2).T
    rhs = np.arange(6, 12, dtype=np.float32).reshape(2, 3).T

    result = atlas.matmul(lhs, rhs)

    assert not lhs.flags.c_contiguous
    assert not rhs.flags.c_contiguous
    np.testing.assert_array_equal(result, np.matmul(lhs, rhs))


def test_matmul_translates_incompatible_shapes_to_shape_error() -> None:
    with pytest.raises(atlas.ShapeError, match="shape mismatch for matmul"):
        atlas.matmul(np.zeros((2, 3)), np.zeros((4, 2)))
