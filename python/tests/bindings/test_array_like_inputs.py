import numpy as np
import pytest

import atlas


@pytest.mark.parametrize("values", [[3, 1, 2], (3, 1, 2), range(3)])
def test_single_array_operations_accept_python_sequences(values: object) -> None:
    np.testing.assert_array_equal(atlas.sort(values), np.sort(np.asarray(values)))


def test_binary_operations_coerce_sequence_operands() -> None:
    np.testing.assert_array_equal(atlas.add([1, 2], (3, 4)), np.array([4, 6]))


def test_array_collections_coerce_each_sequence() -> None:
    np.testing.assert_array_equal(
        atlas.concatenate([[1, 2], (3, 4)], axis=0), np.array([1, 2, 3, 4])
    )


def test_scalar_inputs_remain_invalid_for_array_arguments() -> None:
    with pytest.raises(TypeError, match="expected a NumPy ndarray or Python sequence"):
        atlas.sort(1)
