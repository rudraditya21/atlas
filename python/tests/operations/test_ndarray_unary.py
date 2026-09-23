import numpy as np

import atlas


def test_unary_operations_support_signed_integers() -> None:
    values = np.array([-3, 0, 2], dtype=np.int32)

    assert atlas.neg(values).dtype == values.dtype
    assert atlas.neg(values).tolist() == [3, 0, -2]
    assert atlas.abs(values).tolist() == [3, 0, 2]
    assert atlas.sign(values).tolist() == [-1, 0, 1]
    assert atlas.round(values).tolist() == [-3, 0, 2]


def test_unary_operations_handle_floats_nan_and_infinity() -> None:
    values = np.array([-1.6, 0.2, np.nan, np.inf, -np.inf], dtype=np.float64)

    np.testing.assert_allclose(
        atlas.neg(values),
        np.array([1.6, -0.2, np.nan, -np.inf, np.inf]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        atlas.abs(values), np.array([1.6, 0.2, np.nan, np.inf, np.inf]), equal_nan=True
    )
    np.testing.assert_allclose(
        atlas.sign(values), np.array([-1.0, 1.0, np.nan, 1.0, -1.0]), equal_nan=True
    )
    np.testing.assert_allclose(
        atlas.round(values),
        np.array([-2.0, 0.0, np.nan, np.inf, -np.inf]),
        equal_nan=True,
    )
    assert atlas.isnan(values).tolist() == [False, False, True, False, False]
    assert atlas.isinf(values).tolist() == [False, False, False, True, True]
    assert atlas.isfinite(values).tolist() == [True, True, False, False, False]


def test_unary_operations_preserve_logical_values_from_non_contiguous_inputs() -> None:
    values = np.array([[-1.2, 2.2], [-3.4, 4.4]], dtype=np.float32).T

    result = atlas.round(values)

    assert not values.flags.c_contiguous
    assert result.flags.c_contiguous
    assert result.dtype == values.dtype
    assert result.tolist() == [[-1.0, -3.0], [2.0, 4.0]]
    assert atlas.isfinite(values).tolist() == [[True, True], [True, True]]
