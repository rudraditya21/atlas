import numpy as np
import pytest

import atlas


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, np.bool_),
        (-128, np.int8),
        (127, np.uint8),
        (np.uint16(256), np.uint16),
        (1.5, np.float32),
        (np.float16(1.5), np.float32),
    ],
)
def test_min_scalar_type_returns_the_smallest_supported_dtype(
    value: object, expected: type[np.generic]
) -> None:
    assert atlas.min_scalar_type(value) == np.dtype(expected)


@pytest.mark.parametrize("value", [1 + 2j, "atlas"])
def test_min_scalar_type_rejects_unsupported_dtypes(value: object) -> None:
    with pytest.raises(TypeError, match="unsupported scalar dtype"):
        atlas.min_scalar_type(value)
