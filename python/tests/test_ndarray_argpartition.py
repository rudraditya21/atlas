import numpy as np
import pytest

import atlas


@pytest.mark.parametrize(
    "dtype",
    [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    ],
)
def test_argpartition_returns_indices_for_supported_dtypes(dtype: str) -> None:
    values = np.array([3, 1, 2, 0], dtype=dtype)
    indices = atlas.argpartition(values, 2)
    partitioned = values[indices]

    assert indices.dtype == np.dtype("int64")
    np.testing.assert_array_equal(np.sort(indices), np.arange(values.size))
    assert np.all(partitioned[:2] <= partitioned[2])
    assert np.all(partitioned[3:] >= partitioned[2])


def test_argpartition_uses_logical_view_values() -> None:
    values = np.array([[9, 1, 8], [2, 7, 3]], dtype=np.int64).T
    indices = atlas.argpartition(values, 1, axis=0)
    partitioned = np.take_along_axis(values, indices, axis=0)

    assert not values.flags.c_contiguous
    np.testing.assert_array_equal(partitioned[1], np.sort(values, axis=0)[1])
    assert np.all(partitioned[:1] <= partitioned[1:2])
    assert np.all(partitioned[2:] >= partitioned[1:2])


def test_argpartition_supports_flattened_axis() -> None:
    values = np.array([[9, 1, 8], [2, 7, 3]], dtype=np.int64).T
    indices = atlas.argpartition(values, 2, axis=None)
    partitioned = values.ravel()[indices]

    assert not values.flags.c_contiguous
    assert indices.shape == (values.size,)
    np.testing.assert_array_equal(np.sort(indices), np.arange(values.size))
    assert partitioned[2] == 3
    assert np.all(partitioned[:2] <= partitioned[2])
    assert np.all(partitioned[3:] >= partitioned[2])


def test_argpartition_uses_argsort_nan_ordering() -> None:
    values = np.array([np.nan, 2.0, 1.0, np.nan, 0.0], dtype=np.float64)
    indices = atlas.argpartition(values, 3)
    partitioned = values[indices]
    keys = np.where(np.isnan(partitioned), np.inf, partitioned)

    assert np.isnan(partitioned[3])
    assert np.all(keys[:3] <= keys[3])
    assert np.all(keys[4:] >= keys[3])
    assert np.isnan(partitioned).sum() == 2
