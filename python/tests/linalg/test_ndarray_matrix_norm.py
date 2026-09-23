import math

import numpy as np
import pytest

import atlas


@pytest.mark.parametrize(
    ("order", "expected"),
    [("fro", math.sqrt(30.0)), ("l1", 6.0), ("inf", 7.0)],
)
def test_matrix_norm_supports_all_orders(order: str, expected: float) -> None:
    values = np.array([[1.0, -2.0], [3.0, -4.0]])

    assert atlas.matrix_norm(values, order) == pytest.approx(expected)


def test_matrix_norm_supports_non_contiguous_views_and_zero_matrices() -> None:
    values = np.array([[1.0, -2.0], [3.0, -4.0]]).T

    assert not values.flags.c_contiguous
    assert atlas.matrix_norm(values, "l1") == pytest.approx(
        np.linalg.norm(values, ord=1)
    )
    assert atlas.matrix_norm(np.zeros((2, 3)), "inf") == 0.0


@pytest.mark.parametrize("values", [np.array([1.0, 2.0]), np.array(1.0)])
def test_matrix_norm_translates_invalid_ranks_to_shape_errors(
    values: np.ndarray,
) -> None:
    with pytest.raises(atlas.ShapeError, match="invalid input rank for matrix_norm"):
        atlas.matrix_norm(values)


def test_matrix_norm_rejects_invalid_orders() -> None:
    with pytest.raises(ValueError, match="matrix norm order"):
        atlas.matrix_norm(np.eye(2), "l2")
