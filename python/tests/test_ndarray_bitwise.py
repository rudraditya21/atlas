import numpy as np

import atlas


def test_bitwise_operations_support_signed_integer_arrays_and_scalars() -> None:
    values = np.array([12, 10], dtype=np.int32)

    assert atlas.bitwise_and(values, 6).tolist() == [4, 2]
    assert atlas.bitwise_or(values, 6).tolist() == [14, 14]
    assert atlas.bitwise_xor(values, 6).tolist() == [10, 12]
    assert atlas.bitwise_not(values).tolist() == [-13, -11]


def test_bitwise_operations_support_unsigned_integer_arrays() -> None:
    lhs = np.array([12, 10], dtype=np.uint8)
    rhs = np.array([6, 3], dtype=np.uint8)

    assert atlas.bitwise_and(lhs, rhs).tolist() == [4, 2]
    assert atlas.bitwise_or(lhs, rhs).tolist() == [14, 11]
    assert atlas.bitwise_xor(lhs, rhs).tolist() == [10, 9]
    assert atlas.bitwise_not(lhs).tolist() == [243, 245]


def test_bitwise_operations_support_boolean_arrays() -> None:
    lhs = np.array([True, False], dtype=bool)
    rhs = np.array([True, True], dtype=bool)

    assert atlas.bitwise_and(lhs, rhs).tolist() == [True, False]
    assert atlas.bitwise_or(lhs, rhs).tolist() == [True, True]
    assert atlas.bitwise_xor(lhs, rhs).tolist() == [False, True]
    assert atlas.bitwise_not(lhs).tolist() == [False, True]
    assert atlas.bitwise_and(lhs, True).tolist() == [True, False]


def test_bitwise_operations_support_non_contiguous_views() -> None:
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64).T

    result = atlas.bitwise_xor(values, 1)

    assert not values.flags.c_contiguous
    np.testing.assert_array_equal(result, np.bitwise_xor(values, 1))
