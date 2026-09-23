import numpy as np

import atlas


def test_shares_memory_distinguishes_views_and_copies() -> None:
    source = np.arange(6, dtype=np.int32).reshape(2, 3)
    view = source[:, 1:]
    copied = atlas.copy(source)

    assert atlas.shares_memory(source, view)
    assert not atlas.shares_memory(source, copied)


def test_may_share_memory_recognizes_broadcast_views() -> None:
    source = np.array([1, 2, 3], dtype=np.int32)
    broadcast = atlas.broadcast_to(source, (2, 3))

    assert atlas.may_share_memory(source, broadcast)
    assert atlas.shares_memory(source, broadcast)


def test_memory_helpers_return_python_booleans_for_array_like_inputs() -> None:
    result = atlas.shares_memory([1, 2], [1, 2])

    assert isinstance(result, bool)
    assert not result
