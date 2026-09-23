import numpy as np
import pytest

import atlas


def test_inverse_supports_invertible_matrices() -> None:
    values = np.array([[4.0, 7.0], [2.0, 6.0]], dtype=np.float64)

    result = atlas.inverse(values)

    assert result.dtype == values.dtype
    np.testing.assert_allclose(result, np.linalg.inv(values))


def test_inverse_supports_non_contiguous_views() -> None:
    values = np.array([[4.0, 2.0], [7.0, 6.0]], dtype=np.float32).T

    assert not values.flags.c_contiguous
    np.testing.assert_allclose(atlas.inverse(values), np.linalg.inv(values), rtol=1e-6)


def test_inverse_rejects_singular_matrices() -> None:
    with pytest.raises(atlas.NumericError, match="singular matrix for lu"):
        atlas.inverse(np.array([[1.0, 2.0], [2.0, 4.0]]))


@pytest.mark.parametrize("values", [np.ones((2, 3)), np.array([1.0, 2.0])])
def test_inverse_translates_invalid_shapes_to_shape_errors(values: np.ndarray) -> None:
    with pytest.raises(atlas.ShapeError):
        atlas.inverse(values)
