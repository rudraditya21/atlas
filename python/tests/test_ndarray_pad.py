import numpy as np
import pytest

import atlas


def test_pad_supports_scalar_and_per_axis_widths() -> None:
    value = np.array([[1, 2], [3, 4]], dtype=np.int32)

    np.testing.assert_array_equal(atlas.pad(value, 1), np.pad(value, 1))
    np.testing.assert_array_equal(
        atlas.pad(value, [(1, 0), (2, 1)], value=-1),
        np.pad(value, [(1, 0), (2, 1)], constant_values=-1),
    )


def test_pad_supports_numeric_dtypes_and_empty_dimensions() -> None:
    value = np.empty((2, 0), dtype=np.float32)

    result = atlas.pad(value, [(1, 1), (2, 3)], value=1.5)

    assert result.shape == (4, 5)
    assert result.dtype == np.dtype(np.float32)
    assert result[0, 0] == np.float32(1.5)


@pytest.mark.parametrize("widths", [[(1, -1)], [(1, 1), (2, 2)]])
def test_pad_rejects_invalid_widths(widths: list[tuple[int, int]]) -> None:
    with pytest.raises(atlas.NumericError if widths == [(1, -1)] else atlas.ShapeError):
        atlas.pad(np.array([1, 2]), widths)
