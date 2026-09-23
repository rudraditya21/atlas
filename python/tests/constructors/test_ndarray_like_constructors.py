import numpy as np

import atlas


def test_like_constructors_preserve_shape_and_dtype() -> None:
    source = np.arange(6, dtype=np.int32).reshape(2, 3).T

    zeros = atlas.zeros_like(source)
    ones = atlas.ones_like(source)
    full = atlas.full_like(source, 7)

    for result in (zeros, ones, full):
        assert result.shape == source.shape
        assert result.dtype == source.dtype
    np.testing.assert_array_equal(zeros, np.zeros_like(source))
    np.testing.assert_array_equal(ones, np.ones_like(source))
    np.testing.assert_array_equal(full, np.full_like(source, 7))


def test_like_constructors_accept_dtype_overrides_and_sequences() -> None:
    source = [[1, 2], [3, 4]]

    zeros = atlas.zeros_like(source, dtype="float32")
    full = atlas.full_like(source, 1.5, dtype=np.float64)

    assert zeros.dtype == np.dtype(np.float32)
    assert full.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(zeros, [[0.0, 0.0], [0.0, 0.0]])
    np.testing.assert_array_equal(full, [[1.5, 1.5], [1.5, 1.5]])
