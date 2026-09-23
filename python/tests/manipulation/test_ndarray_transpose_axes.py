import numpy as np
import pytest

import atlas


def test_transpose_supports_explicit_and_negative_axis_permutations() -> None:
    values = np.arange(24, dtype=np.int32).reshape(2, 3, 4)

    expected = np.transpose(values, (2, 0, 1))

    np.testing.assert_array_equal(atlas.transpose(values, [2, 0, 1]), expected)
    np.testing.assert_array_equal(atlas.transpose(values, [-1, -3, -2]), expected)


def test_transpose_preserves_zero_sized_dimensions_with_explicit_axes() -> None:
    values = np.empty((2, 0, 3), dtype=np.float64)

    result = atlas.transpose(values, [2, 0, 1])

    assert result.shape == (3, 2, 0)
    assert result.dtype == values.dtype


def test_transpose_rejects_duplicate_and_incomplete_permutations() -> None:
    values = np.ones((2, 3, 4))

    with pytest.raises(atlas.ShapeError, match="axes must be a permutation"):
        atlas.transpose(values, [0, 0, 1])

    with pytest.raises(atlas.ShapeError, match="dimension mismatch"):
        atlas.transpose(values, [0, 1])


def test_transpose_translates_out_of_range_axes() -> None:
    with pytest.raises(atlas.AxisError, match="invalid axis"):
        atlas.transpose(np.ones((2, 3, 4)), [0, 1, 3])
