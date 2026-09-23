import numpy as np
import pytest

import atlas


def test_asarray_returns_a_dtype_compatible_ndarray_without_copying() -> None:
    source = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).T
    result = atlas.asarray(source)

    assert result is source
    assert result.dtype == np.dtype(np.float32)
    assert result.tolist() == [[1.0, 3.0], [2.0, 4.0]]


def test_asarray_copies_only_when_dtype_conversion_is_required() -> None:
    source = np.array([1, 2], dtype=np.int32)
    result = atlas.asarray(source, dtype="float64")

    assert result is not source
    assert result.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(result, source)


@pytest.mark.parametrize(
    ("constructor", "shape", "dtype", "expected"),
    [
        (atlas.zeros, [], "float64", 0.0),
        (atlas.ones, [2, 0, 3], "int64", [[], []]),
        (atlas.zeros, [2, 2], "float32", [[0.0, 0.0], [0.0, 0.0]]),
    ],
)
def test_zeros_and_ones_support_shapes_and_dtypes(
    constructor: object, shape: list[int], dtype: str, expected: object
) -> None:
    result = constructor(shape, dtype=dtype)

    assert result.dtype == np.dtype(dtype)
    assert result.tolist() == expected


def test_full_supports_boolean_dtype() -> None:
    result = atlas.full([2, 2], True, dtype="bool")

    assert result.dtype == np.dtype(bool)
    assert result.tolist() == [[True, True], [True, True]]


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    ],
)
def test_eye_supports_numeric_dtypes(dtype: str) -> None:
    result = atlas.eye(2, 3, dtype=dtype)

    assert result.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(result, np.eye(2, 3, dtype=dtype))


def test_eye_defaults_to_a_square_float64_matrix_and_supports_empty_dimensions() -> (
    None
):
    result = atlas.eye(3)
    empty = atlas.eye(0, 2, dtype="int64")

    assert result.dtype == np.dtype("float64")
    np.testing.assert_array_equal(result, np.eye(3))
    assert empty.dtype == np.dtype("int64")
    assert empty.shape == (0, 2)


def test_eye_rejects_boolean_dtype() -> None:
    with pytest.raises(ValueError, match="does not support bool"):
        atlas.eye(2, dtype="bool")


@pytest.mark.parametrize("dtype", ["int32", "uint64", "float32", "float64"])
def test_identity_supports_numeric_dtypes(dtype: str) -> None:
    result = atlas.identity(3, dtype=dtype)

    assert result.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(result, np.identity(3, dtype=dtype))


def test_identity_defaults_to_a_square_float64_matrix_and_supports_zero_size() -> None:
    result = atlas.identity(2)
    empty = atlas.identity(0, dtype="int64")

    assert result.dtype == np.dtype("float64")
    np.testing.assert_array_equal(result, np.identity(2))
    assert empty.dtype == np.dtype("int64")
    assert empty.shape == (0, 0)


def test_arange_supports_default_and_selected_dtypes() -> None:
    assert atlas.arange(4).tolist() == [0.0, 1.0, 2.0, 3.0]
    assert atlas.arange(1, 5, 2, dtype="int64").tolist() == [1, 3]


def test_range_constructor_defaults_and_endpoints() -> None:
    arange = atlas.arange(0.0, 1.0, 0.25)
    linspace = atlas.linspace(0.0, 1.0, 5)

    assert arange.dtype == np.dtype("float64")
    np.testing.assert_array_equal(arange, np.array([0.0, 0.25, 0.5, 0.75]))
    assert linspace.dtype == np.dtype("float64")
    np.testing.assert_array_equal(linspace, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))


def test_linspace_supports_float_dtypes() -> None:
    default = atlas.linspace(-1.0, 1.0, 5)
    float32 = atlas.linspace(-1.0, 1.0, 5, dtype="float32")
    single = atlas.linspace(2.5, 9.0, 1, dtype="float32")
    empty = atlas.linspace(0.0, 1.0, 0, dtype="float64")

    assert default.dtype == np.dtype("float64")
    np.testing.assert_array_equal(default, np.linspace(-1.0, 1.0, 5))
    assert float32.dtype == np.dtype("float32")
    np.testing.assert_array_equal(float32, np.linspace(-1.0, 1.0, 5, dtype=np.float32))
    assert single.dtype == np.dtype("float32")
    np.testing.assert_array_equal(single, np.array([2.5], dtype=np.float32))
    assert empty.dtype == np.dtype("float64")
    assert empty.shape == (0,)


@pytest.mark.parametrize(
    ("start", "stop", "num"),
    [
        (0.0, 1.0, 5),
        (3.0, -1.0, 4),
        (2.5, 9.0, 1),
        (0.0, 1.0, 0),
    ],
)
def test_linspace_can_exclude_the_endpoint(start: float, stop: float, num: int) -> None:
    result = atlas.linspace(start, stop, num, endpoint=False)

    np.testing.assert_array_equal(result, np.linspace(start, stop, num, endpoint=False))


def test_integer_arange_preserves_values_above_f64_integer_precision() -> None:
    start = 2**53 + 1

    assert atlas.arange(start, start + 4, 1, dtype="int64").tolist() == [
        start,
        start + 1,
        start + 2,
        start + 3,
    ]
    assert atlas.arange(start, start + 4, 1, dtype="uint64").tolist() == [
        start,
        start + 1,
        start + 2,
        start + 3,
    ]


@pytest.mark.parametrize(
    ("args", "dtype"),
    [
        ((0.5, 3.0, 1.0), "int32"),
        ((-1.0, 3.0, 1.0), "uint8"),
        ((0.0, float("inf"), 1.0), "int16"),
        ((0.0, 3.0, float("nan")), "uint16"),
        ((128.0, 129.0, 1.0), "int8"),
    ],
)
def test_integer_arange_rejects_values_outside_the_target_dtype(
    args: tuple[float, float, float], dtype: str
) -> None:
    with pytest.raises(ValueError, match="integer arange"):
        atlas.arange(*args, dtype=dtype)


def test_range_constructors_reject_unsupported_dtypes_and_non_finite_inputs() -> None:
    with pytest.raises(ValueError, match="arange does not support bool"):
        atlas.arange(0, 3, dtype="bool")
    with pytest.raises(ValueError, match="linspace only supports"):
        atlas.linspace(0.0, 1.0, 3, dtype="int64")
    with pytest.raises(atlas.NumericError, match="finite"):
        atlas.arange(0.0, float("inf"))
    with pytest.raises(atlas.NumericError, match="finite"):
        atlas.linspace(0.0, float("inf"), 3)


def test_range_constructors_handle_empty_and_singleton_outputs() -> None:
    arange = atlas.arange(3, 3, dtype="int64")
    empty = atlas.linspace(0.0, 1.0, 0)
    single = atlas.linspace(2.5, 9.0, 1)

    assert arange.dtype == np.dtype("int64")
    assert arange.shape == (0,)
    assert empty.shape == (0,)
    assert single.tolist() == [2.5]


@pytest.mark.parametrize("shape", [[2**64], [2, 2**64]])
def test_constructors_reject_invalid_dimensions(shape: list[int]) -> None:
    with pytest.raises((OverflowError, ValueError)):
        atlas.zeros(shape)
