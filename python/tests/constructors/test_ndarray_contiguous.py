import numpy as np

import atlas


def test_ascontiguousarray_returns_compatible_contiguous_inputs() -> None:
    source = np.array([[1, 2], [3, 4]], dtype=np.int32)

    result = atlas.ascontiguousarray(source)

    assert result is source


def test_ascontiguousarray_materializes_logical_views_and_dtype_conversions() -> None:
    source = np.arange(6, dtype=np.int32).reshape(2, 3).T

    result = atlas.ascontiguousarray(source)
    converted = atlas.ascontiguousarray(source, dtype=np.float64)

    np.testing.assert_array_equal(result, source)
    assert result.flags.c_contiguous
    assert not np.shares_memory(result, source)
    assert converted.flags.c_contiguous
    assert converted.dtype == np.dtype(np.float64)


def test_ascontiguousarray_follows_numpy_scalar_shape_semantics() -> None:
    result = atlas.ascontiguousarray(np.array(7, dtype=np.int32))

    np.testing.assert_array_equal(
        result, np.ascontiguousarray(np.array(7, dtype=np.int32))
    )
