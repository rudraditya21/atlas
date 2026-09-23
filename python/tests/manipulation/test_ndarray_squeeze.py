import numpy as np
import pytest

import atlas


def test_squeeze_removes_all_singleton_dimensions() -> None:
    values = np.arange(6, dtype=np.int32).reshape(1, 2, 1, 3)

    result = atlas.squeeze(values)

    assert result.shape == (2, 3)
    assert result.tolist() == [[0, 1, 2], [3, 4, 5]]


def test_squeeze_supports_explicit_and_negative_axes() -> None:
    values = np.arange(6, dtype=np.float64).reshape(2, 1, 3)

    np.testing.assert_array_equal(atlas.squeeze(values, 1), np.squeeze(values, axis=1))
    np.testing.assert_array_equal(
        atlas.squeeze(values, -2), np.squeeze(values, axis=-2)
    )


def test_squeeze_supports_axis_sequences() -> None:
    values = np.arange(6, dtype=np.float64).reshape(1, 2, 1, 3)

    np.testing.assert_array_equal(
        atlas.squeeze(values, axis=(0, 2)), np.squeeze(values, axis=(0, 2))
    )
    np.testing.assert_array_equal(
        atlas.squeeze(values, axis=(-4, -2)), np.squeeze(values, axis=(-4, -2))
    )


def test_squeeze_supports_scalar_arrays_and_views() -> None:
    scalar = np.array(7, dtype=np.int64)
    values = np.arange(6, dtype=np.float64).reshape(2, 1, 3).transpose(2, 1, 0)

    assert atlas.squeeze(scalar).shape == ()
    assert atlas.squeeze(scalar).item() == 7
    assert not values.flags.c_contiguous
    np.testing.assert_array_equal(atlas.squeeze(values, 1), np.squeeze(values, axis=1))


def test_squeeze_translates_invalid_dimensions_and_axes() -> None:
    with pytest.raises(atlas.ShapeError, match="axis must have length 1"):
        atlas.squeeze(np.ones((2, 3)), 0)

    with pytest.raises(atlas.AxisError, match="invalid axis"):
        atlas.squeeze(np.ones((2, 1)), 2)

    with pytest.raises(atlas.ShapeError, match="axes must be unique"):
        atlas.squeeze(np.ones((1, 1)), (0, 0))
