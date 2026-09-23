import numpy as np
import pytest

import atlas


def test_axis_cumulative_operations_support_positive_and_negative_axes() -> None:
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

    summed = atlas.cumsum_axis(values, 0)
    product = atlas.cumprod_axis(values, -1)

    np.testing.assert_array_equal(atlas.cumsum(values, axis=0), summed)
    np.testing.assert_array_equal(atlas.cumprod(values, axis=-1), product)
    assert summed.dtype == values.dtype
    assert summed.tolist() == [[1, 2, 3], [5, 7, 9]]
    assert product.dtype == values.dtype
    assert product.tolist() == [[1, 2, 6], [4, 20, 120]]


@pytest.mark.parametrize(
    ("operation", "numpy_operation"),
    [(atlas.cumsum_axis, np.cumsum), (atlas.cumprod_axis, np.cumprod)],
)
def test_axis_cumulative_operations_support_transposed_views(
    operation: object, numpy_operation: object
) -> None:
    values = np.arange(1, 7, dtype=np.float64).reshape(2, 3).T

    result = operation(values, 0)

    assert not values.flags.c_contiguous
    np.testing.assert_array_equal(result, numpy_operation(values, axis=0))


@pytest.mark.parametrize("operation", [atlas.cumsum_axis, atlas.cumprod_axis])
def test_axis_cumulative_operations_preserve_zero_sized_dimensions(
    operation: object,
) -> None:
    values = np.empty((2, 0, 3), dtype=np.float32)

    result = operation(values, 1)

    assert result.shape == values.shape
    assert result.dtype == values.dtype
    assert result.size == 0


@pytest.mark.parametrize("operation", [atlas.cumsum_axis, atlas.cumprod_axis])
def test_axis_cumulative_operations_translate_invalid_axes(operation: object) -> None:
    with pytest.raises(atlas.AxisError, match="invalid axis"):
        operation(np.ones((2, 3), dtype=np.float64), 2)
