import numpy as np
import pytest

import atlas


def test_choose_selects_broadcastable_arrays() -> None:
    indices = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int32)
    choices = [
        np.array([10, 20, 30], dtype=np.int32),
        np.array([40, 50, 60], dtype=np.int32),
        np.array([70, 80, 90], dtype=np.int32),
    ]

    result = atlas.choose(indices, choices)

    np.testing.assert_array_equal(result, np.choose(indices, choices))


def test_choose_supports_scalar_choices_and_promotes_dtypes() -> None:
    indices = np.array([0, 1, 0], dtype=np.int8)

    result = atlas.choose(indices, [1, np.array([2.0, 3.0, 4.0])])

    np.testing.assert_array_equal(result, [1.0, 3.0, 1.0])
    assert result.dtype == np.dtype(np.float64)


@pytest.mark.parametrize(
    ("indices", "choices", "error", "message"),
    [
        (np.array([0.0]), [1], TypeError, "integer dtype"),
        (np.array([1]), [1], ValueError, "choices range"),
        (np.array([0]), [], ValueError, "non-empty"),
    ],
)
def test_choose_validates_indices_and_choices(
    indices: np.ndarray, choices: list[object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        atlas.choose(indices, choices)
