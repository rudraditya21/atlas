import numpy as np
import pytest

import atlas


def test_solve_supports_vector_and_matrix_right_hand_sides() -> None:
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
    vector_rhs = np.array([9.0, 8.0])
    matrix_rhs = np.array([[9.0, 1.0], [8.0, 2.0]])

    np.testing.assert_allclose(
        atlas.solve(matrix, vector_rhs), np.linalg.solve(matrix, vector_rhs), atol=1e-12
    )
    np.testing.assert_allclose(
        atlas.solve(matrix, matrix_rhs), np.linalg.solve(matrix, matrix_rhs), atol=1e-12
    )


def test_solve_supports_non_contiguous_inputs() -> None:
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]]).T
    rhs = np.array([[9.0, 8.0], [1.0, 2.0]]).T

    assert not matrix.flags.c_contiguous
    assert not rhs.flags.c_contiguous
    np.testing.assert_allclose(
        atlas.solve(matrix, rhs), np.linalg.solve(matrix, rhs), atol=1e-12
    )


def test_solve_rejects_singular_systems() -> None:
    with pytest.raises(atlas.NumericError, match="singular matrix for lu"):
        atlas.solve(np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([1.0, 2.0]))


def test_solve_translates_incompatible_shapes_to_shape_errors() -> None:
    with pytest.raises(atlas.ShapeError, match="shape mismatch for solve"):
        atlas.solve(np.eye(2), np.array([1.0, 2.0, 3.0]))
