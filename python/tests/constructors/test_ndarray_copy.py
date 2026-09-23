import numpy as np

import atlas


def test_copy_materializes_independent_logical_values() -> None:
    source = np.arange(6, dtype=np.int32).reshape(2, 3).T

    result = atlas.copy(source)

    np.testing.assert_array_equal(result, source)
    assert result.flags.c_contiguous
    assert not np.shares_memory(result, source)

    result[0, 0] = 99
    assert source[0, 0] == 0


def test_copy_accepts_python_sequences() -> None:
    result = atlas.copy([[1, 2], [3, 4]])

    np.testing.assert_array_equal(result, [[1, 2], [3, 4]])
