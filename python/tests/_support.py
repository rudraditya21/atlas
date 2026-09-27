from importlib import import_module
from types import ModuleType

import numpy as np
import pytest


def require_interop_module(name: str) -> ModuleType:
    try:
        return import_module(name)
    except ModuleNotFoundError as error:
        if error.name == name:
            pytest.skip(f"{name} is required for dataframe interop tests")
        raise


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
