import numpy as np
import pytest

import atlas


def test_copyto_copies_broadcastable_sources_in_place() -> None:
    destination = np.zeros((2, 3), dtype=np.int32)

    result = atlas.copyto(destination, [1, 2, 3])

    assert result is None
    np.testing.assert_array_equal(destination, [[1, 2, 3], [1, 2, 3]])


def test_copyto_respects_boolean_where_masks() -> None:
    destination = np.zeros((2, 2), dtype=np.int32)
    mask = np.array([[True, False], [False, True]])

    atlas.copyto(destination, 7, where=mask)

    np.testing.assert_array_equal(destination, [[7, 0], [0, 7]])


def test_copyto_validates_destinations_masks_and_broadcasts() -> None:
    with pytest.raises(TypeError, match="destination"):
        atlas.copyto([0, 0], 1)
    with pytest.raises(TypeError, match="boolean dtype"):
        atlas.copyto(np.zeros(2), 1, where=[1, 0])
    with pytest.raises(atlas.ShapeError, match="broadcast"):
        atlas.copyto(np.zeros((2, 3)), np.ones((2, 4)))
