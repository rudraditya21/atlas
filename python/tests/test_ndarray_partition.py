import numpy as np
import pytest
import atlas


def test_partition_supports_values_views_and_axes() -> None:
    integers = np.array([3, 1, 2, 1], dtype=np.int32)
    view = np.array([[3.0, 1.0], [4.0, 2.0]]).T
    np.testing.assert_array_equal(
        atlas.partition(integers, 2), np.partition(integers, 2)
    )
    assert not view.flags.c_contiguous
    np.testing.assert_array_equal(
        atlas.partition(view, 1, axis=0), np.partition(view, 1, axis=0)
    )


@pytest.mark.parametrize("kth", [-4, 3])
def test_partition_rejects_invalid_kth(kth: int) -> None:
    with pytest.raises(atlas.AxisError, match="index"):
        atlas.partition(np.array([1, 2, 3]), kth)
