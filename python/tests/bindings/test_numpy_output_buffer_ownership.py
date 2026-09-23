import gc
from collections.abc import Callable

import numpy as np
import pytest

import atlas


def constructor_output() -> np.ndarray:
    return atlas.ones([2], dtype="int32")


def arithmetic_output() -> np.ndarray:
    return atlas.add(np.array([1, 2], dtype=np.int32), 1)


def cast_output() -> np.ndarray:
    return atlas.astype(np.array([1, 2], dtype=np.int16), "int32")


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (constructor_output, [1, 1]),
        (arithmetic_output, [2, 3]),
        (cast_output, [1, 2]),
    ],
)
def test_native_outputs_retain_transferred_buffers_after_return(
    factory: Callable[[], np.ndarray], expected: list[int]
) -> None:
    result = factory()

    assert result.base is not None
    gc.collect()
    assert result.tolist() == expected

    result[0] = 9
    assert result[0] == 9
