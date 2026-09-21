import numpy as np
import pytest

import atlas


def test_allclose_applies_absolute_and_relative_tolerances() -> None:
    lhs = np.array([1.0, 1000.0], dtype=np.float64)
    rhs = np.array([1.000001, 1000.01], dtype=np.float64)

    assert atlas.allclose(lhs, rhs)
    assert not atlas.allclose(lhs, rhs, rtol=0.0, atol=0.0)
    assert atlas.allclose(lhs, rhs, rtol=0.0, atol=0.02)


def test_allclose_supports_broadcasting() -> None:
    lhs = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)
    rhs = np.array([1.0, 2.0], dtype=np.float32)

    assert atlas.allclose(lhs, rhs)


def test_allclose_handles_nan_according_to_equal_nan() -> None:
    lhs = np.array([np.nan, 1.0], dtype=np.float64)
    rhs = np.array([np.nan, 1.0], dtype=np.float64)

    assert not atlas.allclose(lhs, rhs)
    assert atlas.allclose(lhs, rhs, equal_nan=True)


@pytest.mark.parametrize(
    "rtol, atol", [(-1.0, 0.0), (0.0, -1.0), (np.inf, 0.0), (0.0, np.nan)]
)
def test_allclose_rejects_invalid_tolerances(rtol: float, atol: float) -> None:
    values = np.array([1.0], dtype=np.float64)

    with pytest.raises(
        atlas.NumericError, match="tolerances must be finite and non-negative"
    ):
        atlas.allclose(values, values, rtol=rtol, atol=atol)


def test_allclose_rejects_mixed_dtypes() -> None:
    with pytest.raises(TypeError, match="unsupported NumPy dtype"):
        atlas.allclose(
            np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float64)
        )


def test_allclose_translates_incompatible_shapes_to_shape_error() -> None:
    with pytest.raises(atlas.ShapeError, match="invalid broadcast"):
        atlas.allclose(
            np.zeros((2, 3), dtype=np.float64), np.zeros((4, 2), dtype=np.float64)
        )
