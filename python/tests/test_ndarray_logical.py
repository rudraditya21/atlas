import numpy as np
import pytest

import atlas


def test_comparisons_support_scalars_broadcasting_and_views() -> None:
    values = np.arange(6.0).reshape(2, 3).T

    assert atlas.greater_equal(values, 3.0).tolist() == [
        [False, True],
        [False, True],
        [False, True],
    ]
    assert atlas.equal(values, np.array([0.0, 4.0])).tolist() == [
        [True, False],
        [False, True],
        [False, False],
    ]


def test_selection_count_nonzero_and_masked_fill_use_logical_values() -> None:
    values = np.arange(6.0).reshape(2, 3).T
    mask = np.array([[True, False], [False, True], [True, False]])

    assert atlas.select(values, mask).tolist() == [0.0, 4.0, 2.0]
    assert atlas.count_true(mask.T.T) == 3
    assert atlas.nonzero(mask.T.T).tolist() == [[0, 0], [1, 1], [2, 0]]
    assert atlas.masked_fill(values, mask, -1.0).tolist() == [
        [-1.0, 3.0],
        [1.0, -1.0],
        [-1.0, 5.0],
    ]
    assert values[0, 0] == 0.0


def test_logical_operations_handle_scalars_and_empty_arrays() -> None:
    assert atlas.select(np.array(2.0), np.array(True)).tolist() == [2.0]
    assert atlas.nonzero(np.empty((2, 0))).shape == (0, 2)
    assert atlas.count_true(np.empty((2, 0), dtype=bool)) == 0


def test_logical_operations_reject_invalid_masks_and_dtypes() -> None:
    with pytest.raises(TypeError, match="unsupported NumPy dtype"):
        atlas.equal(np.array([1], dtype=np.float16), 1)

    with pytest.raises(atlas.NumericError, match="mask"):
        atlas.select(np.ones((2, 2)), np.ones((3, 2), dtype=bool))
