import numpy as np
import pytest

import atlas


def test_public_operations_translate_atlas_errors() -> None:
    with pytest.raises(atlas.ShapeError, match="invalid broadcast"):
        atlas.add(np.zeros((2, 3)), np.zeros((4, 2)))

    with pytest.raises(atlas.NumericError, match="division by zero"):
        atlas.divide(np.array([1], dtype=np.int64), 0)

    with pytest.raises(atlas.ShapeError, match="mask shape mismatch"):
        atlas.select(np.ones((2, 2)), np.ones((4,), dtype=bool))

    with pytest.raises(atlas.ShapeError, match="shape overflow"):
        atlas.zeros([int(np.iinfo(np.uintp).max), 2])
