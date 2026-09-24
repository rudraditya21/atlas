import numpy as np
import pytest

import atlas


@pytest.mark.parametrize("dtype", ["int32", np.dtype("int32"), np.int32])
def test_dtype_utilities_accept_standard_dtype_specifications(dtype: object) -> None:
    assert atlas.result_type(dtype, np.float32) == np.result_type(
        np.dtype(dtype), np.float32
    )
    assert atlas.promote_types(dtype, np.float32) == np.promote_types(
        np.dtype(dtype), np.float32
    )
    assert atlas.can_cast(dtype, np.float64)
    assert atlas.issubdtype(dtype, np.integer)


def test_dtype_utilities_follow_numpy_return_conventions() -> None:
    assert isinstance(atlas.result_type(np.int32, np.float32), np.dtype)
    assert isinstance(atlas.promote_types(np.int32, np.float32), np.dtype)
    assert isinstance(atlas.can_cast(np.int32, np.float32), bool)
    assert isinstance(atlas.issubdtype(np.int32, np.integer), bool)
    assert isinstance(atlas.finfo(np.float32), np.finfo)
    assert isinstance(atlas.iinfo(np.int32), np.iinfo)


@pytest.mark.parametrize(
    ("function", "dtype", "message"),
    [
        (atlas.finfo, "float16", "unsupported dtype"),
        (atlas.iinfo, "float64", "integer dtype"),
        (atlas.finfo, None, "dtype is required"),
    ],
)
def test_dtype_limit_utilities_reject_unsupported_or_ambiguous_dtypes(
    function: object, dtype: object, message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        function(dtype)


@pytest.mark.parametrize(
    ("value", "dtype"),
    [
        (-128, np.int8),
        (-129, np.int16),
        (255, np.uint8),
        (256, np.uint16),
        (np.float16(1.5), np.float32),
    ],
)
def test_min_scalar_type_respects_supported_dtype_boundaries(
    value: object, dtype: type[np.generic]
) -> None:
    assert atlas.min_scalar_type(value) == np.dtype(dtype)
