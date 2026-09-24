import numpy as np

import atlas


def test_result_type_promotes_array_like_inputs() -> None:
    result = atlas.result_type(np.array([1], dtype=np.int32), [1.0])

    assert result == np.result_type(np.array([1], dtype=np.int32), np.array([1.0]))


def test_result_type_accepts_scalars_and_dtype_specifications() -> None:
    result = atlas.result_type(1, np.float32, "int16", np.dtype("uint8"))

    assert result == np.result_type(1, np.float32, np.dtype("int16"), np.dtype("uint8"))
    assert isinstance(result, np.dtype)


def test_result_type_preserves_scalar_value_promotion() -> None:
    values = np.array([1], dtype=np.int8)

    assert atlas.result_type(values, 127) == np.result_type(values, 127)
