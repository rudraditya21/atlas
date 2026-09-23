import numpy as np

import atlas


def test_empty_like_preserves_source_shape_and_dtype() -> None:
    source = np.arange(6, dtype=np.int32).reshape(2, 3).T

    result = atlas.empty_like(source)

    assert result.shape == source.shape
    assert result.dtype == source.dtype


def test_empty_like_accepts_dtype_overrides_and_sequences() -> None:
    result = atlas.empty_like([[1, 2], [3, 4]], dtype=np.float64)

    assert result.shape == (2, 2)
    assert result.dtype == np.dtype(np.float64)
