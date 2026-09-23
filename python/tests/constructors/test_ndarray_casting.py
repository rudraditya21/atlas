import numpy as np
import pytest

import atlas


@pytest.mark.parametrize(
    ("source", "dtype", "expected"),
    [
        (np.array([-2, 3], dtype=np.int32), "float64", [-2.0, 3.0]),
        (np.array([-2, 3], dtype=np.int16), "int32", [-2, 3]),
    ],
)
def test_astype_performs_lossless_casts_and_preserves_output_dtype(
    source: np.ndarray, dtype: str, expected: list[int | float]
) -> None:
    result = atlas.astype(source, dtype)

    assert result.dtype == np.dtype(dtype)
    assert result.tolist() == expected


def test_astype_rejects_lossy_casts() -> None:
    with pytest.raises(atlas.NumericError, match="invalid cast"):
        atlas.astype(np.array([1.5], dtype=np.float64), "int64")
