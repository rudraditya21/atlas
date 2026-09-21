import numpy as np
import pytest

import atlas


def test_take_supports_positive_negative_and_repeated_indices() -> None:
    values = np.arange(6, dtype=np.int32).reshape(2, 3)

    assert atlas.take(values, [2, 0, 2], 1).tolist() == [[2, 0, 2], [5, 3, 5]]
    assert atlas.take(values, [-1, 0], -1).tolist() == [[2, 0], [5, 3]]


def test_take_supports_non_contiguous_views() -> None:
    values = np.arange(6, dtype=np.float64).reshape(2, 3).T

    result = atlas.take(values, [-1, 0], 0)

    assert not values.flags.c_contiguous
    assert result.tolist() == [[2.0, 5.0], [0.0, 3.0]]


def test_take_translates_invalid_indices_to_axis_errors() -> None:
    with pytest.raises(atlas.AxisError, match="index out of bounds on axis 1"):
        atlas.take(np.arange(6).reshape(2, 3), [3], 1)
