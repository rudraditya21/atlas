import numpy as np
import pytest

import atlas


def test_axis_index_reductions_use_first_indices_for_ties() -> None:
    values = np.array([[1, 1, 2], [3, 3, 0]], dtype=np.int32)

    np.testing.assert_array_equal(
        atlas.argmin(values, axis=1), atlas.argmin_axis(values, 1)
    )
    np.testing.assert_array_equal(
        atlas.argmax(values, axis=-1), atlas.argmax_axis(values, -1)
    )
    assert atlas.argmin_axis(values, 1).tolist() == [0, 2]
    assert atlas.argmax_axis(values, -1).tolist() == [2, 0]


@pytest.mark.parametrize("reduction", [atlas.argmin_axis, atlas.argmax_axis])
def test_axis_index_reductions_return_first_nan_indices(reduction: object) -> None:
    values = np.array([[np.nan, 2.0], [1.0, np.nan]])

    assert reduction(values, 1).tolist() == [0, 1]


@pytest.mark.parametrize(
    ("reduction", "numpy_reduction"),
    [(atlas.argmin_axis, np.argmin), (atlas.argmax_axis, np.argmax)],
)
def test_axis_index_reductions_support_transposed_views(
    reduction: object, numpy_reduction: object
) -> None:
    values = np.array([[3, 0, 5], [2, 4, 1]], dtype=np.int64).T

    assert not values.flags.c_contiguous
    np.testing.assert_array_equal(reduction(values, 0), numpy_reduction(values, axis=0))


@pytest.mark.parametrize(
    ("reduction", "numpy_reduction"),
    [(atlas.argmin, np.argmin), (atlas.argmax, np.argmax)],
)
def test_axis_index_reductions_support_keepdims(
    reduction: object, numpy_reduction: object
) -> None:
    values = np.array([[3, 1, 2], [6, 5, 4]], dtype=np.int32)

    np.testing.assert_array_equal(
        reduction(values, axis=1, keepdims=True),
        numpy_reduction(values, axis=1, keepdims=True),
    )


@pytest.mark.parametrize("reduction", [atlas.argmin_axis, atlas.argmax_axis])
def test_axis_index_reductions_translate_invalid_axes(reduction: object) -> None:
    with pytest.raises(atlas.AxisError, match="invalid axis"):
        reduction(np.ones((2, 3), dtype=np.float64), 2)
