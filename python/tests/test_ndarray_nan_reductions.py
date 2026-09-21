import numpy as np
import pytest

import atlas


@pytest.mark.parametrize(
    ("reduction", "expected"),
    [
        (atlas.nanmin, 1.0),
        (atlas.nanmax, 3.0),
        (atlas.nanmean, 2.0),
        (atlas.nanstd, 1.0),
    ],
)
def test_nan_reductions_ignore_nan_values(reduction: object, expected: float) -> None:
    assert reduction(np.array([1.0, np.nan, 3.0])) == expected


@pytest.mark.parametrize(
    "reduction", [atlas.nanmin, atlas.nanmax, atlas.nanmean, atlas.nanstd]
)
def test_nan_reductions_reject_all_nan_inputs(reduction: object) -> None:
    with pytest.raises(atlas.NumericError, match="all values are NaN"):
        reduction(np.array([np.nan, np.nan]))


@pytest.mark.parametrize(
    "reduction", [atlas.nanmin, atlas.nanmax, atlas.nanmean, atlas.nanstd]
)
def test_nan_reductions_reject_empty_arrays(reduction: object) -> None:
    with pytest.raises(atlas.NumericError, match="empty input"):
        reduction(np.array([], dtype=np.float64))


@pytest.mark.parametrize(
    ("reduction", "numpy_reduction"),
    [
        (atlas.nanmin, np.nanmin),
        (atlas.nanmax, np.nanmax),
        (atlas.nanmean, np.nanmean),
        (atlas.nanstd, np.nanstd),
    ],
)
def test_nan_reductions_support_transposed_views(
    reduction: object, numpy_reduction: object
) -> None:
    values = np.array([[np.nan, 4.0], [1.0, 3.0]], dtype=np.float64).T

    assert not values.flags.c_contiguous
    assert reduction(values) == pytest.approx(numpy_reduction(values))
