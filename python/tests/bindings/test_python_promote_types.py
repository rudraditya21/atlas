import numpy as np
import pytest

import atlas


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("int32", np.float32),
        (np.dtype("uint8"), np.dtype("int16")),
        (np.int64, "uint64"),
    ],
)
def test_promote_types_accepts_dtype_specifications(
    left: object, right: object
) -> None:
    assert atlas.promote_types(left, right) == np.promote_types(
        np.dtype(left), np.dtype(right)
    )


def test_promote_types_rejects_scalar_values() -> None:
    with pytest.raises(TypeError):
        atlas.promote_types(1, np.int32)
