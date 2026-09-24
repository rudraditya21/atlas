import numpy as np

import atlas


def test_common_type_promotes_array_like_inputs_to_a_floating_dtype() -> None:
    values = np.array([1, 2], dtype=np.int16)
    floats = [1.0, 2.0]

    assert atlas.common_type(values, floats) == np.common_type(
        values, np.asarray(floats)
    )


def test_common_type_accepts_scalars_and_preserves_numpy_return_type() -> None:
    result = atlas.common_type(np.float32(1.0), np.int32(1))

    assert result == np.common_type(
        np.asarray(np.float32(1.0)), np.asarray(np.int32(1))
    )
    assert isinstance(result, type)
