import numpy as np
import pytest

import atlas


@pytest.mark.parametrize(
    ("array", "shape", "values"),
    [
        (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            [2, 2],
            [1.0, 2.0, 3.0, 4.0],
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]]).T,
            [2, 2],
            [1.0, 3.0, 2.0, 4.0],
        ),
        (
            np.arange(20.0).reshape(4, 5)[1:4:2, 1:5:2],
            [2, 2],
            [6.0, 8.0, 16.0, 18.0],
        ),
        (np.array(7.0), [], [7.0]),
        (np.empty((2, 0, 3)), [2, 0, 3], []),
    ],
)
def test_numpy_arrays_materialize_in_logical_row_major_order(
    array: np.ndarray, shape: list[int], values: list[float]
) -> None:
    actual_shape, actual_values = atlas._native._array_f64_parts(array)

    assert actual_shape == shape
    assert actual_values == values
