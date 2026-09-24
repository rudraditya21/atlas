import numpy as np
import pytest

import atlas


@pytest.mark.parametrize("casting", ["no", "equiv", "safe", "same_kind", "unsafe"])
def test_can_cast_matches_numpy_casting_rules(casting: str) -> None:
    assert atlas.can_cast("int32", "float32", casting=casting) == np.can_cast(
        "int32", "float32", casting=casting
    )


def test_can_cast_accepts_dtype_objects_and_classes() -> None:
    assert atlas.can_cast(np.dtype("uint8"), np.int16)


def test_can_cast_rejects_unknown_casting_rules() -> None:
    with pytest.raises(ValueError):
        atlas.can_cast("int32", "float32", casting="invalid")
