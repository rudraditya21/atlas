import numpy as np

import atlas


def test_matmul_supports_matching_batched_matrices() -> None:
    lhs = np.arange(12, dtype=np.int64).reshape(2, 2, 3)
    rhs = np.arange(12, 24, dtype=np.int64).reshape(2, 3, 2)

    result = atlas.matmul(lhs, rhs)

    assert result.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(result, np.matmul(lhs, rhs))
