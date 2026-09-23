import numpy as np
import pytest
import atlas


def test_partition_supports_values_views_and_axes() -> None:
    integers = np.array([9, 1, 8, 2, 7], dtype=np.int32)
    result = atlas.partition(integers, 2)

    np.testing.assert_array_equal(np.sort(result), np.sort(integers))
    assert np.all(result[:2] <= result[2])
    assert np.all(result[3:] >= result[2])

    view = np.array([[9.0, 1.0, 8.0], [2.0, 7.0, 3.0]]).T
    assert not view.flags.c_contiguous
    result = atlas.partition(view, 1, axis=0)
    expected = np.sort(view, axis=0)[1]
    np.testing.assert_array_equal(result[1], expected)
    assert np.all(result[:1] <= result[1:2])
    assert np.all(result[2:] >= result[1:2])


def test_partition_supports_negative_kth_and_axes() -> None:
    values = np.array([[9, 1, 8], [2, 7, 3]], dtype=np.int64)
    result = atlas.partition(values, 1, axis=-1)

    np.testing.assert_array_equal(result[:, 1], np.sort(values, axis=-1)[:, 1])
    assert np.all(result[:, :1] <= result[:, 1:2])
    assert np.all(result[:, 2:] >= result[:, 1:2])

    result = atlas.partition(values[0], -1)
    assert result[-1] == np.max(values[0])
    assert np.all(result[:-1] <= result[-1])


def test_partition_supports_flattened_axis() -> None:
    values = np.array([[9, 1, 8], [2, 7, 3]], dtype=np.int64).T
    result = atlas.partition(values, 2, axis=None)

    assert not values.flags.c_contiguous
    assert result.shape == (values.size,)
    assert result[2] == 3
    assert np.all(result[:2] <= result[2])
    assert np.all(result[3:] >= result[2])
    np.testing.assert_array_equal(np.sort(result), np.sort(values, axis=None))


def test_partition_supports_multiple_kth_values() -> None:
    values = np.array([9, 1, 8, 2, 7], dtype=np.int64)
    result = atlas.partition(values, [1, 3])

    np.testing.assert_array_equal(result[[1, 3]], np.sort(values)[[1, 3]])
    assert np.all(result[:1] <= result[1])
    assert np.all((result[2:3] >= result[1]) & (result[2:3] <= result[3]))
    assert np.all(result[4:] >= result[3])


@pytest.mark.parametrize(
    ("kth", "exception", "match"),
    [
        ([], atlas.NumericError, "must not be empty"),
        ([2, 1], atlas.NumericError, "ordered and unique"),
        ([1, 1], atlas.NumericError, "ordered and unique"),
        ([3], atlas.AxisError, "index"),
    ],
)
def test_partition_rejects_invalid_multiple_kth_values(
    kth: list[int], exception: type[Exception], match: str
) -> None:
    with pytest.raises(exception, match=match):
        atlas.partition(np.array([1, 2, 3]), kth)


def test_partition_handles_duplicate_and_nan_values() -> None:
    duplicates = np.array([3, 1, 2, 2, 2, 4], dtype=np.int32)
    result = atlas.partition(duplicates, 3)

    assert result[3] == 2
    assert np.all(result[:3] <= result[3])
    assert np.all(result[4:] >= result[3])
    np.testing.assert_array_equal(np.sort(result), np.sort(duplicates))

    values = np.array([np.nan, 2.0, 1.0, np.nan, 0.0], dtype=np.float64)
    result = atlas.partition(values, 3)

    assert np.isnan(result[3])
    assert not np.isnan(result[:3]).any()
    assert np.isnan(result).sum() == 2
    np.testing.assert_array_equal(np.sort(result), np.sort(values))


def test_partition_rejects_empty_partition_axes() -> None:
    with pytest.raises(atlas.AxisError, match="index"):
        atlas.partition(np.empty((2, 0), dtype=np.float64), 0, axis=1)


@pytest.mark.parametrize("kth", [-4, 3])
def test_partition_rejects_invalid_kth(kth: int) -> None:
    with pytest.raises(atlas.AxisError, match="index"):
        atlas.partition(np.array([1, 2, 3]), kth)
