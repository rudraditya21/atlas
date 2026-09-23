import numpy as np
import pytest

import atlas


@pytest.mark.parametrize(
    "array",
    [
        np.array(1.0, dtype=np.float64),
        np.array([], dtype=np.float32),
        np.array([True, False], dtype=bool),
        np.array([[1, 2], [3, 4]], dtype=np.int64),
        np.zeros((2, 3, 4), dtype=np.float64),
    ],
)
def test_ndarray_metadata_matches_numpy(array: np.ndarray) -> None:
    assert atlas.shape(array) == array.shape
    assert atlas.ndim(array) == array.ndim
    assert atlas.size(array) == array.size
    assert atlas.dtype(array) == array.dtype
