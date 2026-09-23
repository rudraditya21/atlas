import numpy as np
import pytest

import atlas


def test_split_returns_partitions_at_requested_indices() -> None:
    value = np.arange(12, dtype=np.int32).reshape(3, 4)

    result = atlas.split(value, [1, 3], axis=1)

    assert isinstance(result, list)
    assert [part.dtype for part in result] == [np.dtype(np.int32)] * 3
    np.testing.assert_array_equal(result[0], [[0], [4], [8]])
    np.testing.assert_array_equal(result[1], [[1, 2], [5, 6], [9, 10]])
    np.testing.assert_array_equal(result[2], [[3], [7], [11]])


def test_split_accepts_an_equal_section_count() -> None:
    value = np.arange(12, dtype=np.int32).reshape(3, 4)

    result = atlas.split(value, 2, axis=1)

    assert len(result) == 2
    np.testing.assert_array_equal(result[0], value[:, :2])
    np.testing.assert_array_equal(result[1], value[:, 2:])


def test_split_supports_negative_axes_and_empty_partitions() -> None:
    value = np.arange(6, dtype=np.float64).reshape(2, 3)

    result = atlas.split(value, [0, 2, 2], axis=-1)

    assert [part.shape for part in result] == [(2, 0), (2, 2), (2, 0), (2, 1)]
    np.testing.assert_array_equal(result[1], [[0.0, 1.0], [3.0, 4.0]])


def test_split_uses_logical_values_from_views() -> None:
    value = np.arange(12, dtype=np.int64).reshape(3, 4).T

    assert not value.flags.c_contiguous
    result = atlas.split(value, [1, 3], axis=0)

    np.testing.assert_array_equal(result[0], value[:1])
    np.testing.assert_array_equal(result[1], value[1:3])
    np.testing.assert_array_equal(result[2], value[3:])


@pytest.mark.parametrize("indices", [[3, 2], [5]])
def test_split_rejects_invalid_indices(indices: list[int]) -> None:
    with pytest.raises(atlas.NumericError, match="split indices"):
        atlas.split(np.arange(4), indices, axis=0)


def test_split_rejects_non_divisible_section_counts() -> None:
    with pytest.raises(atlas.NumericError, match="divisible by sections"):
        atlas.split(np.arange(5), 2, axis=0)
