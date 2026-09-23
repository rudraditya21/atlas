import numpy as np
import pytest

import atlas


def test_searchsorted_supports_scalar_and_vector_values() -> None:
    sorted_values = np.array([1, 3, 5, 7], dtype=np.int32)

    assert atlas.searchsorted(sorted_values, 4) == 2
    np.testing.assert_array_equal(
        atlas.searchsorted(sorted_values, np.array([0, 4, 8], dtype=np.int32)),
        [0, 2, 4],
    )


def test_searchsorted_supports_left_right_duplicates_and_views() -> None:
    sorted_values = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)[::2]

    assert not sorted_values.flags.c_contiguous
    assert atlas.searchsorted(sorted_values, 2.0, side="left") == 1
    assert atlas.searchsorted(sorted_values, 2.0, side="right") == 2


def test_searchsorted_supports_sorter_indices() -> None:
    values = np.array([30, 10, 40, 20], dtype=np.int32)
    sorter = np.array([1, 3, 0, 2], dtype=np.int64)

    np.testing.assert_array_equal(
        atlas.searchsorted(values, [15, 35], sorter=sorter),
        np.searchsorted(values, [15, 35], sorter=sorter),
    )


@pytest.mark.parametrize(
    ("sorter", "error", "message"),
    [
        (np.array([[0, 1], [2, 3]], dtype=np.int64), ValueError, "one-dimensional"),
        (np.array([0.0, 1.0, 2.0, 3.0]), TypeError, "integer dtype"),
        (np.array([0, 1, 1, 3], dtype=np.int64), ValueError, "permutation"),
        (np.array([0, 1, 2, 4], dtype=np.int64), ValueError, "in bounds"),
        (np.array([0, 1, 2], dtype=np.int64), ValueError, "same length"),
    ],
)
def test_searchsorted_validates_sorter_indices(
    sorter: np.ndarray, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        atlas.searchsorted(np.array([10, 20, 30, 40]), 25, sorter=sorter)


def test_searchsorted_rejects_invalid_dtypes() -> None:
    with pytest.raises(TypeError, match="unsupported NumPy dtype"):
        atlas.searchsorted(np.array([1], dtype=np.float16), 1)
