import numpy as np

import atlas


def test_put_assigns_scalar_values_at_flattened_indices() -> None:
    values = np.arange(6, dtype=np.int32).reshape(2, 3)

    result = atlas.put(values, [0, 4], 9)

    assert result is None
    np.testing.assert_array_equal(values, [[9, 1, 2], [3, 9, 5]])


def test_put_repeats_array_values_and_keeps_last_duplicate_write() -> None:
    values = np.zeros(5, dtype=np.int32)

    atlas.put(values, [1, 3, 1, 4], [2, 4])

    np.testing.assert_array_equal(values, [0, 2, 0, 4, 4])


def test_put_assigns_non_contiguous_views_in_logical_order() -> None:
    source = np.arange(6, dtype=np.int32).reshape(2, 3).T
    expected = source.copy()

    atlas.put(source, [0, 5], [10, 20])
    np.put(expected, [0, 5], [10, 20])

    np.testing.assert_array_equal(source, expected)
