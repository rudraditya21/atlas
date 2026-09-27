import numpy as np


def assert_numpy_array(
    actual: np.ndarray,
    expected: object,
    *,
    dtype: object,
    shape: tuple[int, ...] | None = None,
) -> None:
    expected_array = np.asarray(expected, dtype=dtype)

    assert actual.dtype == np.dtype(dtype)
    assert actual.shape == (expected_array.shape if shape is None else shape)
    np.testing.assert_array_equal(actual, expected_array)
