import numpy as np
import pytest

import atlas


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [(np.dtype("float32"), np.dtype("float32")), (np.float64, np.dtype("float64"))],
)
def test_dtype_objects_and_scalar_classes_work_in_every_dtype_api(
    dtype: object, expected: np.dtype
) -> None:
    source = np.array([1, 2], dtype=expected)

    assert atlas.asarray(source, dtype=dtype).dtype == expected
    assert atlas.zeros([2], dtype=dtype).dtype == expected
    assert atlas.ones([2], dtype=dtype).dtype == expected
    assert atlas.full([2], 1, dtype=dtype).dtype == expected
    assert atlas.eye(2, dtype=dtype).dtype == expected
    assert atlas.identity(2, dtype=dtype).dtype == expected
    assert atlas.arange(2, dtype=dtype).dtype == expected
    assert atlas.linspace(0.0, 1.0, 2, dtype=dtype).dtype == expected
    assert atlas.astype(source, dtype).dtype == expected


def test_integer_scalar_dtype_classes_remain_supported() -> None:
    dtype = np.int32

    assert atlas.zeros([2], dtype=dtype).dtype == np.dtype(dtype)
    assert atlas.arange(2, dtype=dtype).dtype == np.dtype(dtype)
    assert atlas.astype(np.array([1], dtype=np.int64), dtype).dtype == np.dtype(dtype)


def test_unsupported_numpy_dtype_specifications_report_value_errors() -> None:
    with pytest.raises(ValueError, match="unsupported dtype"):
        atlas.zeros([2], dtype=np.float16)
