import numpy as np
import pytest

import atlas


def test_reshape_and_transpose_preserve_logical_values() -> None:
    values = np.arange(6, dtype=np.int32)

    reshaped = atlas.reshape(values, [2, 3])
    transposed = atlas.transpose(reshaped)

    assert reshaped.dtype == np.dtype(np.int32)
    assert reshaped.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert transposed.tolist() == [[0, 3], [1, 4], [2, 5]]


def test_shape_operations_support_zero_sized_dimensions() -> None:
    values = np.empty((2, 0, 3), dtype=np.float64)

    assert atlas.reshape(values, [0]).shape == (0,)
    assert atlas.transpose(values).shape == (3, 0, 2)


def test_reshape_translates_invalid_shapes_to_shape_errors() -> None:
    with pytest.raises(atlas.ShapeError, match="invalid reshape"):
        atlas.reshape(np.arange(6), [5])

    with pytest.raises(atlas.ShapeError, match="shape overflow"):
        atlas.reshape(np.array(1), [int(np.iinfo(np.uintp).max), 2])
