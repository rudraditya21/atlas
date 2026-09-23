import numpy as np
import pytest

import atlas


def test_tile_supports_scalar_and_vector_repetitions() -> None:
    value = np.array([[1, 2]], dtype=np.int32)

    np.testing.assert_array_equal(atlas.tile(value, 2), np.tile(value, 2))
    np.testing.assert_array_equal(atlas.tile(value, [2, 1]), np.tile(value, [2, 1]))


def test_tile_uses_logical_values_from_views() -> None:
    value = np.arange(6, dtype=np.float64).reshape(2, 3).T

    assert not value.flags.c_contiguous
    np.testing.assert_array_equal(atlas.tile(value, [1, 2]), np.tile(value, [1, 2]))


def test_tile_preserves_zero_sized_dimensions() -> None:
    value = np.empty((2, 0, 3), dtype=np.uint8)

    result = atlas.tile(value, [2, 3, 4])

    assert result.shape == (4, 0, 12)
    assert result.dtype == np.dtype(np.uint8)


@pytest.mark.parametrize("repetitions", [[], [-1], -1])
def test_tile_rejects_invalid_repetitions(repetitions: list[int] | int) -> None:
    with pytest.raises(atlas.NumericError, match="repetitions"):
        atlas.tile(np.array([1, 2]), repetitions)
