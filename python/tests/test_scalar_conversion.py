import numpy as np
import pytest

import atlas


@pytest.mark.parametrize(
    ("value", "expected_kind"),
    [
        (True, "bool"),
        (-7, "int64"),
        (2**63, "uint64"),
        (np.int32(-7), "int32"),
        (1.5, "float64"),
        (np.float32(1.5), "float32"),
        (np.float64(1.5), "float64"),
    ],
)
def test_scalar_conversion_preserves_supported_kinds(
    value: object, expected_kind: str
) -> None:
    assert atlas._native._scalar_kind(value) == expected_kind


def test_scalar_conversion_rejects_integer_overflow() -> None:
    with pytest.raises(OverflowError, match="outside Atlas's supported range"):
        atlas._native._scalar_kind(2**64)


def test_scalar_conversion_rejects_complex_numpy_scalars() -> None:
    with pytest.raises(TypeError, match="unsupported NumPy scalar dtype"):
        atlas._native._scalar_kind(np.complex128(1 + 2j))


def test_scalar_conversion_rejects_object_values() -> None:
    with pytest.raises(TypeError, match="expected a bool, integer, float"):
        atlas._native._scalar_kind(object())
