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


def test_take_without_an_axis_flattens_logical_values() -> None:
    values = np.arange(6, dtype=np.float64).reshape(2, 3).T

    result = atlas.take(values, [-1, 0, 2], axis=None)

    assert not values.flags.c_contiguous
    np.testing.assert_array_equal(result, np.take(values, [-1, 0, 2], axis=None))


@pytest.mark.parametrize("mode", ["wrap", "clip"])
def test_take_supports_index_modes(mode: str) -> None:
    values = np.arange(6, dtype=np.int32).reshape(2, 3)
    indices = [-4, -1, 3, 4]

    np.testing.assert_array_equal(
        atlas.take(values, indices, axis=1, mode=mode),
        np.take(values, indices, axis=1, mode=mode),
    )


def test_take_wraps_flattened_logical_values() -> None:
    values = np.arange(6, dtype=np.int32).reshape(2, 3).T

    np.testing.assert_array_equal(
        atlas.take(values, [-7, 6], axis=None, mode="wrap"),
        np.take(values, [-7, 6], axis=None, mode="wrap"),
    )


def test_take_rejects_unknown_index_modes() -> None:
    with pytest.raises(ValueError, match="mode must"):
        atlas.take(np.arange(3), [0], mode="invalid")


def test_take_translates_invalid_indices_to_axis_errors() -> None:
    with pytest.raises(atlas.AxisError, match="index out of bounds on axis 1"):
        atlas.take(np.arange(6).reshape(2, 3), [3], 1)
