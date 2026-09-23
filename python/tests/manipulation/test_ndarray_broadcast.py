import numpy as np
import pytest

import atlas


def test_broadcast_to_returns_a_read_only_logical_view() -> None:
    values = np.array([1, 2, 3], dtype=np.int32)

    result = atlas.broadcast_to(values, (2, 3))

    np.testing.assert_array_equal(result, [[1, 2, 3], [1, 2, 3]])
    assert not result.flags.writeable
    assert np.shares_memory(result, values)


def test_broadcast_to_supports_scalars_and_views() -> None:
    scalar = np.array(7, dtype=np.float64)
    values = np.arange(6, dtype=np.int32).reshape(2, 3).T[:, :1]

    np.testing.assert_array_equal(
        atlas.broadcast_to(scalar, (2, 2)), [[7.0, 7.0], [7.0, 7.0]]
    )
    np.testing.assert_array_equal(
        atlas.broadcast_to(values, (3, 2)), np.broadcast_to(values, (3, 2))
    )


def test_broadcast_to_translates_incompatible_shapes() -> None:
    with pytest.raises(atlas.ShapeError, match="operands could not be broadcast"):
        atlas.broadcast_to(np.ones((2, 3)), (2, 4))


def test_broadcast_arrays_returns_shared_logical_views() -> None:
    rows = np.array([1, 2, 3], dtype=np.int32)
    columns = np.array([[10], [20]], dtype=np.int32)

    result_rows, result_columns = atlas.broadcast_arrays(rows, columns)

    np.testing.assert_array_equal(result_rows, [[1, 2, 3], [1, 2, 3]])
    np.testing.assert_array_equal(result_columns, [[10, 10, 10], [20, 20, 20]])
    assert np.shares_memory(result_rows, rows)
    assert np.shares_memory(result_columns, columns)


def test_broadcast_arrays_accepts_sequences_and_scalars() -> None:
    values, scalar = atlas.broadcast_arrays([1, 2, 3], 4)

    np.testing.assert_array_equal(values, [1, 2, 3])
    np.testing.assert_array_equal(scalar, [4, 4, 4])


def test_broadcast_arrays_translates_incompatible_shapes() -> None:
    with pytest.raises(atlas.ShapeError, match="broadcast"):
        atlas.broadcast_arrays(np.ones((2, 3)), np.ones((2, 4)))
