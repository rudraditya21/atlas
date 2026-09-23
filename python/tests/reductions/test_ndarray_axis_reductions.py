import numpy as np
import pytest

import atlas


def test_axis_reductions_support_positive_and_negative_axes_with_expected_dtypes() -> (
    None
):
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

    summed = atlas.sum_axis(values, 0)
    mean = atlas.mean_axis(values, -1)

    np.testing.assert_array_equal(atlas.sum(values, axis=0), summed)
    np.testing.assert_array_equal(atlas.mean(values, axis=-1), mean)
    assert summed.dtype == np.dtype(np.int32)
    assert summed.tolist() == [5, 7, 9]
    assert mean.dtype == np.dtype(np.float64)
    assert mean.tolist() == [2.0, 5.0]


def test_axis_reductions_support_transposed_views() -> None:
    values = np.arange(6.0).reshape(2, 3).T

    assert not values.flags.c_contiguous
    np.testing.assert_array_equal(atlas.sum_axis(values, -1), np.sum(values, axis=-1))
    np.testing.assert_array_equal(atlas.mean_axis(values, 0), np.mean(values, axis=0))


@pytest.mark.parametrize(
    ("reduction", "numpy_reduction"),
    [
        (atlas.sum, np.sum),
        (atlas.mean, np.mean),
        (atlas.min, np.min),
        (atlas.max, np.max),
    ],
)
def test_axis_reductions_support_keepdims(
    reduction: object, numpy_reduction: object
) -> None:
    values = np.arange(6, dtype=np.int32).reshape(2, 3)

    np.testing.assert_array_equal(
        reduction(values, axis=1, keepdims=True),
        numpy_reduction(values, axis=1, keepdims=True),
    )


@pytest.mark.parametrize("reduction", [atlas.sum_axis, atlas.mean_axis])
def test_axis_reductions_reject_empty_reduction_lanes(reduction: object) -> None:
    with pytest.raises(atlas.NumericError, match="empty input"):
        reduction(np.empty((2, 0, 3), dtype=np.float64), 1)


@pytest.mark.parametrize("reduction", [atlas.sum_axis, atlas.mean_axis])
def test_axis_reductions_translate_invalid_axes(reduction: object) -> None:
    with pytest.raises(atlas.AxisError, match="invalid axis"):
        reduction(np.ones((2, 3), dtype=np.float64), 2)
