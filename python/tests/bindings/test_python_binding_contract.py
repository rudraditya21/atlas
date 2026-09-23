from collections.abc import Callable

import numpy as np
import pytest

import atlas


@pytest.mark.parametrize("shape", [[2, 3], (2, 3)])
def test_constructor_sequence_shapes_and_keywords_are_supported(
    shape: list[int] | tuple[int, int],
) -> None:
    zeros = atlas.zeros(shape=shape, dtype="int32")
    full = atlas.full(shape=shape, fill_value=7, dtype="int32")

    assert zeros.shape == (2, 3)
    assert zeros.dtype == np.dtype("int32")
    assert full.shape == (2, 3)
    assert full.dtype == np.dtype("int32")
    assert np.all(full == 7)


def test_constructor_defaults_match_the_public_contract() -> None:
    eye = atlas.eye(rows=2)
    identity = atlas.identity(size=2)
    linspace = atlas.linspace(start=0.0, stop=1.0, num=3)

    assert eye.dtype == np.dtype("float64")
    np.testing.assert_array_equal(eye, np.eye(2))
    assert identity.dtype == np.dtype("float64")
    np.testing.assert_array_equal(identity, np.identity(2))
    np.testing.assert_array_equal(linspace, np.array([0.0, 0.5, 1.0]))


def test_axis_defaults_and_keywords_match_the_public_contract() -> None:
    values = np.array([[3, 1, 2], [0, 2, 1]], dtype=np.int64)

    np.testing.assert_array_equal(atlas.sort(values), np.sort(values, axis=-1))
    np.testing.assert_array_equal(
        atlas.sort(value=values, axis=0), np.sort(values, axis=0)
    )
    np.testing.assert_array_equal(
        atlas.linspace(0.0, 1.0, 4, endpoint=False),
        np.array([0.0, 0.25, 0.5, 0.75]),
    )


def test_binding_return_types_follow_python_and_numpy_conventions() -> None:
    values = np.array([[1, 2], [3, 4]], dtype=np.int64)

    assert isinstance(atlas.asarray(values), np.ndarray)
    assert atlas.shape(values) == (2, 2)
    assert isinstance(atlas.ndim(values), int)
    assert isinstance(atlas.size(values), int)
    assert atlas.dtype(values) == np.dtype("int64")
    assert isinstance(atlas.searchsorted(np.array([1, 3], dtype=np.int64), 2), int)
    assert isinstance(atlas.all(np.array([True, True])), bool)


@pytest.mark.parametrize(
    ("operation", "exception", "match"),
    [
        (
            lambda: atlas.sort(3),
            TypeError,
            "expected a NumPy ndarray or Python sequence",
        ),
        (lambda: atlas.zeros([2], dtype="complex128"), ValueError, "unsupported dtype"),
        (lambda: atlas.sort(np.array([3, 1, 2]), axis=1), atlas.AxisError, "axis"),
        (lambda: atlas.repeat(np.array([1, 2]), -1), atlas.NumericError, "nonnegative"),
    ],
)
def test_binding_errors_use_the_public_exception_contract(
    operation: Callable[[], object], exception: type[Exception], match: str
) -> None:
    with pytest.raises(exception, match=match):
        operation()
