"""Shared Python binding argument coercion helpers."""

from collections.abc import Sequence
from functools import wraps

import numpy as np

from .errors import AxisError, ShapeError


_ARRAY_LIKE_ERROR = "expected a NumPy ndarray or Python sequence"


def _array_like(value):
    if isinstance(value, np.ndarray) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    ):
        return np.asarray(value)
    raise TypeError(_ARRAY_LIKE_ERROR)


def _is_array_like(value):
    return isinstance(value, np.ndarray) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    )


def _shape_error(error):
    return ShapeError(f"shape mismatch: {error}")


def _axis_error(axis, *, expected):
    return AxisError(f"unsupported axis {axis!r}; expected {expected}")


def _dtype_error(dtype):
    return TypeError(f"unsupported dtype: {dtype}")


def _optional_array_like(value):
    return _array_like(value) if _is_array_like(value) else value


def _array_or_scalar(value):
    return _array_like(value) if _is_array_like(value) else np.asarray(value)


def _document(function, description):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    wrapper.__doc__ = description
    return wrapper


def _coerce_arrays(function, *, required=(), optional=()):
    """Coerce array-like public arguments before entering the native boundary."""

    @wraps(function)
    def wrapper(*args, **kwargs):
        args = list(args)
        kwargs = dict(kwargs)
        for index, name in required:
            if index < len(args):
                args[index] = _array_like(args[index])
            elif name in kwargs:
                kwargs[name] = _array_like(kwargs[name])
        for index, name in optional:
            if index < len(args):
                args[index] = _optional_array_like(args[index])
            elif name in kwargs:
                kwargs[name] = _optional_array_like(kwargs[name])
        return function(*args, **kwargs)

    return wrapper


def _coerce_binary_operands(function):
    """Coerce array-like operands without performing dtype promotion."""

    @wraps(function)
    def wrapper(lhs, rhs):
        lhs_value = _array_like(lhs) if _is_array_like(lhs) else lhs
        rhs_value = _array_like(rhs) if _is_array_like(rhs) else rhs
        if not isinstance(lhs_value, np.ndarray):
            if isinstance(lhs_value, (bool, int)) and isinstance(rhs_value, np.ndarray):
                lhs_value = np.asarray(lhs_value, dtype=rhs_value.dtype)
            else:
                lhs_value = np.asarray(lhs_value)
        if isinstance(rhs_value, bool) and lhs_value.dtype != np.dtype(bool):
            rhs_value = np.asarray(rhs_value, dtype=lhs_value.dtype)
        elif isinstance(rhs_value, int) and lhs_value.dtype == np.dtype(bool):
            rhs_value = np.asarray(rhs_value)
        elif isinstance(rhs_value, np.generic):
            rhs_value = np.asarray(rhs_value)
        elif isinstance(rhs_value, float) and not np.issubdtype(
            lhs_value.dtype, np.floating
        ):
            rhs_value = np.asarray(rhs_value)
        return function(lhs_value, rhs_value)

    return wrapper


def _coerce_weighted_operands(function):
    @wraps(function)
    def wrapper(values, weights):
        values = _array_like(values)
        weights = _array_like(weights)
        dtype = np.result_type(values, weights)
        return function(
            np.asarray(values, dtype=dtype), np.asarray(weights, dtype=dtype)
        )

    return wrapper


def _coerce_array_collection(function):
    @wraps(function)
    def wrapper(arrays, axis=0):
        if isinstance(arrays, (str, bytes, bytearray)):
            raise TypeError("expected an iterable of array-like values")
        try:
            arrays = iter(arrays)
        except TypeError:
            raise TypeError("expected an iterable of array-like values") from None
        return function([_array_like(value) for value in arrays], axis)

    return wrapper


def _coerce_searchsorted(function):
    def wrapper(sorted, values, side="left", *, sorter=None):
        sorted = _array_like(sorted)
        values = (
            np.asarray(values, dtype=sorted.dtype)
            if _is_array_like(values) and not isinstance(values, np.ndarray)
            else _optional_array_like(values)
        )
        return function(
            sorted,
            values,
            side,
            None if sorter is None else _array_like(sorter),
        )

    wrapper.__name__ = function.__name__
    wrapper.__doc__ = function.__doc__
    return wrapper
