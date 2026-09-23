import math

import numpy as np
import pytest

import atlas


def test_axis_min_max_reduce_along_positive_and_negative_axes() -> None:
    values = np.array([[3, 1, 2], [6, 5, 4]], dtype=np.int32)

    minimum = atlas.min_axis(values, 0)
    maximum = atlas.max_axis(values, -1)

    np.testing.assert_array_equal(atlas.min(values, axis=0), minimum)
    np.testing.assert_array_equal(atlas.max(values, axis=-1), maximum)
    assert minimum.dtype == values.dtype
    assert minimum.tolist() == [3, 1, 2]
    assert maximum.dtype == values.dtype
    assert maximum.tolist() == [3, 6]


@pytest.mark.parametrize(
    ("reduction", "expected"), [(atlas.min_axis, 3.0), (atlas.max_axis, 4.0)]
)
def test_axis_min_max_propagate_nan(reduction: object, expected: float) -> None:
    result = reduction(np.array([[1.0, math.nan], [3.0, 4.0]]), 1)

    assert math.isnan(result[0])
    assert result[1] == expected


@pytest.mark.parametrize("reduction", [atlas.min_axis, atlas.max_axis])
def test_axis_min_max_translate_invalid_axes(reduction: object) -> None:
    with pytest.raises(atlas.AxisError, match="invalid axis"):
        reduction(np.ones((2, 3), dtype=np.float64), -3)
