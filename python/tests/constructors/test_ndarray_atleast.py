import numpy as np

import atlas


def test_atleast_helpers_promote_single_inputs_to_the_requested_rank() -> None:
    scalar = np.array(7, dtype=np.int32)
    vector = np.array([1, 2], dtype=np.int32)

    np.testing.assert_array_equal(atlas.atleast_1d(scalar), np.atleast_1d(scalar))
    np.testing.assert_array_equal(atlas.atleast_2d(vector), np.atleast_2d(vector))
    np.testing.assert_array_equal(atlas.atleast_3d(vector), np.atleast_3d(vector))


def test_atleast_helpers_return_tuples_for_multiple_array_like_inputs() -> None:
    result = atlas.atleast_2d([1, 2], np.array(3, dtype=np.int32))

    assert isinstance(result, tuple)
    assert len(result) == 2
    np.testing.assert_array_equal(result[0], [[1, 2]])
    np.testing.assert_array_equal(result[1], [[3]])


def test_atleast_helpers_accept_scalar_inputs() -> None:
    result = atlas.atleast_3d(1.5)

    np.testing.assert_array_equal(result, np.atleast_3d(1.5))
