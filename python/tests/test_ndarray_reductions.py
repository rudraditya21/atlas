import math

import numpy as np
import pytest

import atlas


def test_reductions_support_contiguous_arrays_and_transposed_views() -> None:
    contiguous = np.array([[1.0, 2.0], [3.0, 4.0]])
    transposed = np.arange(6.0).reshape(2, 3).T

    assert atlas.sum(contiguous) == 10.0
    assert atlas.mean(contiguous) == 2.5
    assert atlas.min(contiguous) == 1.0
    assert atlas.max(contiguous) == 4.0
    assert atlas.sum(transposed) == 15.0
    assert atlas.mean(transposed) == 2.5
    assert atlas.min(transposed) == 0.0
    assert atlas.max(transposed) == 5.0


def test_reductions_support_scalar_arrays() -> None:
    value = np.array(7.0)

    assert atlas.sum(value) == 7.0
    assert atlas.mean(value) == 7.0
    assert atlas.min(value) == 7.0
    assert atlas.max(value) == 7.0


@pytest.mark.parametrize("reduction", [atlas.sum, atlas.mean, atlas.min, atlas.max])
def test_reductions_reject_empty_arrays(reduction: object) -> None:
    with pytest.raises(atlas.NumericError, match="empty input"):
        reduction(np.array([], dtype=np.float64))


@pytest.mark.parametrize("reduction", [atlas.sum, atlas.mean, atlas.min, atlas.max])
def test_float_reductions_propagate_nan(reduction: object) -> None:
    assert math.isnan(reduction(np.array([1.0, math.nan, 3.0])))
