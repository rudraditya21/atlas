import numpy as np
import pytest

import atlas


def test_finfo_and_iinfo_return_supported_dtype_limits() -> None:
    float_info = atlas.finfo("float32")
    integer_info = atlas.iinfo(np.uint8)

    assert float_info.max == np.finfo(np.float32).max
    assert integer_info.min == 0
    assert integer_info.max == 255


@pytest.mark.parametrize(
    ("function", "dtype", "message"),
    [
        (atlas.finfo, np.int32, "floating-point"),
        (atlas.iinfo, np.float64, "integer"),
        (atlas.finfo, np.float16, "unsupported dtype"),
        (atlas.iinfo, None, "dtype is required"),
    ],
)
def test_dtype_limit_helpers_reject_unsupported_or_ambiguous_dtypes(
    function: object, dtype: object, message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        function(dtype)
