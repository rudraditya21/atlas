import math

import numpy as np
import pytest

import atlas


@pytest.mark.parametrize(
    "values",
    [np.array([1, 2, 3, 4], dtype=np.int32), np.array([1.0, 2.0, 3.0, 4.0])],
)
def test_statistical_reductions_support_integer_and_float_arrays(
    values: np.ndarray,
) -> None:
    assert atlas.variance(values) == 1.25
    assert atlas.stddev(values) == pytest.approx(math.sqrt(1.25))


def test_statistical_reductions_support_transposed_views() -> None:
    values = np.arange(6.0).reshape(2, 3).T

    assert atlas.variance(values) == pytest.approx(35.0 / 12.0)
    assert atlas.stddev(values) == pytest.approx(math.sqrt(35.0 / 12.0))


@pytest.mark.parametrize("reduction", [atlas.variance, atlas.stddev])
def test_statistical_reductions_reject_empty_arrays(reduction: object) -> None:
    with pytest.raises(atlas.NumericError, match="empty input"):
        reduction(np.array([], dtype=np.float64))


@pytest.mark.parametrize("reduction", [atlas.variance, atlas.stddev])
def test_statistical_reductions_propagate_nan(reduction: object) -> None:
    assert math.isnan(reduction(np.array([1.0, math.nan, 3.0])))
