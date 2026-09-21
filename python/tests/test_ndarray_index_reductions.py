import math

import numpy as np
import pytest

import atlas


def test_index_reductions_support_integer_and_float_arrays() -> None:
    integers = np.array([4, -2, 7, 3], dtype=np.int32)
    floats = np.array([4.5, -2.0, 7.0, 3.0], dtype=np.float64)

    assert atlas.argmin(integers) == 1
    assert atlas.argmax(integers) == 2
    assert atlas.argmin(floats) == 1
    assert atlas.argmax(floats) == 2


@pytest.mark.parametrize("reduction", [atlas.argmin, atlas.argmax])
def test_index_reductions_return_the_first_nan_index(reduction: object) -> None:
    values = np.array([4.0, math.nan, 1.0, math.nan])

    assert reduction(values) == 1


def test_index_reductions_follow_logical_order_for_transposed_views() -> None:
    values = np.array([[3, 0, 5], [2, 4, 1]], dtype=np.int64).T

    assert not values.flags.c_contiguous
    assert atlas.argmin(values) == 2
    assert atlas.argmax(values) == 4


@pytest.mark.parametrize("reduction", [atlas.argmin, atlas.argmax])
def test_index_reductions_reject_empty_arrays(reduction: object) -> None:
    with pytest.raises(atlas.NumericError, match="empty input"):
        reduction(np.array([], dtype=np.float64))
