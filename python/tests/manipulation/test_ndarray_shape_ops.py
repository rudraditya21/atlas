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


def test_reshape_infers_a_single_negative_dimension() -> None:
    values = np.arange(12, dtype=np.int32)

    np.testing.assert_array_equal(atlas.reshape(values, [3, -1]), values.reshape(3, 4))


@pytest.mark.parametrize("shape", [[-1, -1], [5, -1], [-2], [-1, 0]])
def test_reshape_rejects_invalid_inferred_dimensions(shape: list[int]) -> None:
    with pytest.raises(atlas.ShapeError):
        atlas.reshape(np.arange(6), shape)


def test_shape_operations_support_zero_sized_dimensions() -> None:
    values = np.empty((2, 0, 3), dtype=np.float64)

    assert atlas.reshape(values, [0]).shape == (0,)
    assert atlas.transpose(values).shape == (3, 0, 2)


def test_reshape_translates_invalid_shapes_to_shape_errors() -> None:
    with pytest.raises(atlas.ShapeError, match="invalid reshape"):
        atlas.reshape(np.arange(6), [5])

    with pytest.raises(atlas.ShapeError, match="shape overflow"):
        atlas.reshape(np.array(1), [int(np.iinfo(np.uintp).max), 2])
