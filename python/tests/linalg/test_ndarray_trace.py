import numpy as np
import pytest

import atlas


def test_trace_supports_square_and_rectangular_matrices() -> None:
    assert atlas.trace(np.array([[1, 2], [3, 4]], dtype=np.int32)) == 5
    assert atlas.trace(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)) == 6.0


def test_trace_supports_non_contiguous_matrices() -> None:
    values = np.array([[1, 2], [3, 4]], dtype=np.int64).T

    assert not values.flags.c_contiguous
    assert atlas.trace(values) == 5


@pytest.mark.parametrize("values", [np.array([1, 2]), np.array(1)])
def test_trace_translates_invalid_ranks_to_shape_errors(values: np.ndarray) -> None:
    with pytest.raises(atlas.ShapeError, match="invalid input rank for trace"):
        atlas.trace(values)
