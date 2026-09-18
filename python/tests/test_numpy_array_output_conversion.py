import numpy as np
import pytest

import atlas


@pytest.mark.parametrize(
    ("shape", "data", "expected"),
    [
        ([], [7.0], 7.0),
        ([0], [], []),
        ([3], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        ([2, 2], [1.0, 2.0, 3.0, 4.0], [[1.0, 2.0], [3.0, 4.0]]),
        (
            [2, 2, 2],
            list(range(8)),
            [[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]],
        ),
    ],
)
def test_atlas_arrays_convert_to_contiguous_numpy_outputs(
    shape: list[int], data: list[float], expected: object
) -> None:
    result = atlas._native._array_f64_output(shape, data)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.dtype(np.float64)
    assert result.flags.c_contiguous
    assert result.tolist() == expected
