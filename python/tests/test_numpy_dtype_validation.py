import numpy as np
import pytest

import atlas


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (np.array([object()], dtype=object), "unsupported NumPy object dtype"),
        (np.array(["atlas"]), "unsupported NumPy string dtype"),
        (
            np.array([(1, 2.0)], dtype=[("id", "i4"), ("value", "f8")]),
            "unsupported NumPy structured dtype",
        ),
        (np.array([1 + 2j]), "unsupported NumPy complex dtype"),
        (np.array([1], dtype=np.int32), "unsupported NumPy dtype int32"),
    ],
)
def test_array_conversion_rejects_unsupported_dtypes(
    array: np.ndarray, message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        atlas._native._array_f64_parts(array)
