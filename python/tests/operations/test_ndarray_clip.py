import numpy as np
import pytest

import atlas


@pytest.mark.parametrize(
    ("values", "minimum", "maximum", "expected"),
    [
        (np.array([-3, 0, 5], dtype=np.int32), -1, 3, [-1, 0, 3]),
        (np.array([-3.0, 0.5, 5.0], dtype=np.float64), -1.0, 3.0, [-1.0, 0.5, 3.0]),
    ],
)
def test_clip_applies_bounds_and_preserves_dtype(
    values: np.ndarray,
    minimum: int | float,
    maximum: int | float,
    expected: list[int | float],
) -> None:
    result = atlas.clip(values, minimum, maximum)

    assert result.dtype == values.dtype
    assert result.tolist() == expected


def test_clip_supports_non_contiguous_inputs() -> None:
    values = np.array([[-3.0, 1.0], [5.0, 2.0]], dtype=np.float32).T

    result = atlas.clip(values, 0.0, 3.0)

    assert not values.flags.c_contiguous
    assert result.flags.c_contiguous
    assert result.tolist() == [[0.0, 3.0], [1.0, 2.0]]


def test_clip_accepts_either_optional_bound() -> None:
    values = np.array([-3.0, 0.5, 5.0], dtype=np.float64)

    np.testing.assert_array_equal(
        atlas.clip(values, minimum=0.0, maximum=None), np.clip(values, 0.0, None)
    )
    np.testing.assert_array_equal(
        atlas.clip(values, minimum=None, maximum=3.0), np.clip(values, None, 3.0)
    )


def test_clip_rejects_missing_bounds() -> None:
    with pytest.raises(atlas.NumericError, match="at least one bound"):
        atlas.clip(np.array([1.0]), minimum=None, maximum=None)


def test_clip_rejects_reversed_bounds() -> None:
    with pytest.raises(
        atlas.NumericError, match="min must be less than or equal to max"
    ):
        atlas.clip(np.array([1.0]), 2.0, 1.0)
