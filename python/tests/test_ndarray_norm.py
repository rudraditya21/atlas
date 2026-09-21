import math

import numpy as np
import pytest

import atlas


def test_norm_supports_vectors_and_matrices() -> None:
    assert atlas.norm(np.array([3, 4], dtype=np.int32)) == 5.0
    assert atlas.norm(np.array([[1.0, 2.0], [3.0, 4.0]])) == pytest.approx(
        math.sqrt(30.0)
    )


def test_norm_returns_zero_for_zero_arrays() -> None:
    assert atlas.norm(np.zeros((2, 3), dtype=np.float64)) == 0.0


def test_norm_supports_non_contiguous_views() -> None:
    values = np.arange(6.0).reshape(2, 3).T

    assert not values.flags.c_contiguous
    assert atlas.norm(values) == pytest.approx(np.linalg.norm(values))


@pytest.mark.parametrize("values", [np.array(1.0), np.ones((1, 1, 1))])
def test_norm_translates_invalid_ranks_to_shape_errors(values: np.ndarray) -> None:
    with pytest.raises(atlas.ShapeError, match="invalid input rank for norm"):
        atlas.norm(values)
