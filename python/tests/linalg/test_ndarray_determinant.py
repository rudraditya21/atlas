import numpy as np
import pytest

import atlas


@pytest.mark.parametrize(
    "values",
    [
        np.array([[1, 2], [3, 4]], dtype=np.int32),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
    ],
)
def test_det_supports_integer_and_float_matrices(values: np.ndarray) -> None:
    assert atlas.det(values) == pytest.approx(np.linalg.det(values))


def test_det_supports_non_contiguous_views() -> None:
    values = np.array([[1.0, 3.0], [2.0, 4.0]]).T

    assert not values.flags.c_contiguous
    assert atlas.det(values) == pytest.approx(np.linalg.det(values))


def test_det_rejects_singular_matrices() -> None:
    with pytest.raises(atlas.NumericError, match="singular matrix for lu"):
        atlas.det(np.array([[1.0, 2.0], [2.0, 4.0]]))


@pytest.mark.parametrize("values", [np.ones((2, 3)), np.array([1.0, 2.0])])
def test_det_translates_invalid_shapes_to_shape_errors(values: np.ndarray) -> None:
    with pytest.raises(atlas.ShapeError):
        atlas.det(values)
