import numpy as np
import pytest

import atlas


def test_axis_boolean_reductions_support_positive_and_negative_axes() -> None:
    values = np.array([[True, False], [True, True]], dtype=bool)

    assert atlas.all(values, axis=0).tolist() == [True, False]
    assert atlas.any(values, axis=-1).tolist() == [True, True]
    assert atlas.all_axis(values, 0).tolist() == [True, False]
    assert atlas.any_axis(values, -1).tolist() == [True, True]


def test_axis_boolean_reductions_use_empty_lane_identity_values() -> None:
    values = np.empty((2, 0), dtype=bool)

    assert atlas.all_axis(values, 1).tolist() == [True, True]
    assert atlas.any_axis(values, 1).tolist() == [False, False]


@pytest.mark.parametrize("reduction", [atlas.all_axis, atlas.any_axis])
def test_axis_boolean_reductions_translate_invalid_axes(reduction: object) -> None:
    with pytest.raises(atlas.AxisError, match="invalid axis"):
        reduction(np.ones((2, 3), dtype=bool), 2)
