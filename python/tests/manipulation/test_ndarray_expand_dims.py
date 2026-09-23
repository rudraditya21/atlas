import numpy as np
import pytest

import atlas


def test_expand_dims_inserts_dimensions_at_requested_axes() -> None:
    values = np.arange(6, dtype=np.int32).reshape(2, 3)

    np.testing.assert_array_equal(
        atlas.expand_dims(values, 0), np.expand_dims(values, 0)
    )
    np.testing.assert_array_equal(
        atlas.expand_dims(values, 1), np.expand_dims(values, 1)
    )


def test_expand_dims_supports_negative_axes() -> None:
    values = np.arange(6, dtype=np.float64).reshape(2, 3)

    np.testing.assert_array_equal(
        atlas.expand_dims(values, -1), np.expand_dims(values, -1)
    )
    np.testing.assert_array_equal(
        atlas.expand_dims(values, -3), np.expand_dims(values, -3)
    )


def test_expand_dims_supports_scalar_arrays() -> None:
    result = atlas.expand_dims(np.array(7, dtype=np.int64), 0)

    assert result.shape == (1,)
    assert result.tolist() == [7]


@pytest.mark.parametrize("axis", [3, -4])
def test_expand_dims_translates_invalid_axes(axis: int) -> None:
    with pytest.raises(atlas.AxisError, match="invalid axis"):
        atlas.expand_dims(np.ones((2, 3)), axis)
