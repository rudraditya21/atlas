import numpy as np
import pytest

import atlas


def test_diag_constructs_matrices_from_vectors_and_extracts_square_diagonals() -> None:
    assert atlas.diag(np.array([1, 2, 3], dtype=np.int32)).tolist() == [
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3],
    ]
    assert atlas.diag(np.array([[1.0, 2.0], [3.0, 4.0]])).tolist() == [1.0, 4.0]


def test_diag_supports_views_and_rectangular_matrices() -> None:
    values = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64).T
    rectangular = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)

    assert not values.flags.c_contiguous
    assert atlas.diag(values).tolist() == [1, 4]
    assert atlas.diag(rectangular).tolist() == [1, 5]


@pytest.mark.parametrize("values", [np.array(1), np.ones((1, 1, 1))])
def test_diag_translates_invalid_ranks_to_shape_errors(values: np.ndarray) -> None:
    with pytest.raises(atlas.ShapeError, match="invalid input rank for diag"):
        atlas.diag(values)
