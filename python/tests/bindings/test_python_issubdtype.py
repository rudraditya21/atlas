import numpy as np

import atlas


def test_issubdtype_supports_standard_dtype_hierarchies() -> None:
    assert atlas.issubdtype("int32", np.integer)
    assert atlas.issubdtype(np.uint64, np.unsignedinteger)
    assert atlas.issubdtype(np.dtype("float32"), np.floating)
    assert atlas.issubdtype(np.float64, np.number)
    assert atlas.issubdtype(np.bool_, np.bool_)


def test_issubdtype_rejects_unrelated_hierarchies() -> None:
    assert not atlas.issubdtype(np.int32, np.floating)
    assert not atlas.issubdtype(np.bool_, np.number)
