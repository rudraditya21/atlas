import numpy as np
import pytest
import atlas


def test_partition_supports_values_views_and_axes() -> None:
    integers = np.array([9, 1, 8, 2, 7], dtype=np.int32)
    result = atlas.partition(integers, 2)

    np.testing.assert_array_equal(np.sort(result), np.sort(integers))
    assert np.all(result[:2] <= result[2])
    assert np.all(result[3:] >= result[2])

    view = np.array([[9.0, 1.0, 8.0], [2.0, 7.0, 3.0]]).T
    assert not view.flags.c_contiguous
    result = atlas.partition(view, 1, axis=0)
    expected = np.sort(view, axis=0)[1]
    np.testing.assert_array_equal(result[1], expected)
    assert np.all(result[:1] <= result[1:2])
    assert np.all(result[2:] >= result[1:2])


@pytest.mark.parametrize("kth", [-4, 3])
def test_partition_rejects_invalid_kth(kth: int) -> None:
    with pytest.raises(atlas.AxisError, match="index"):
        atlas.partition(np.array([1, 2, 3]), kth)
