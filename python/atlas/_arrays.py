"""NumPy-style array operations exposed by Atlas."""

from collections.abc import Sequence
from operator import index as integer_index

import numpy as np

from . import _native
from ._support import (
    _array_like,
    _array_or_scalar,
    _axis_error,
    _coerce_array_collection,
    _coerce_arrays,
    _coerce_binary_operands,
    _coerce_searchsorted,
    _dtype_error,
    _is_array_like,
    _optional_array_like,
    _shape_error,
)

_ATLAS_DTYPE_NAMES = frozenset(
    {
        "bool",
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
    }
)
_UNSET = object()

asarray = _native.asarray
zeros = _native.zeros
ones = _native.ones
eye = _native.eye
identity = _native.identity
arange = _native.arange
allclose = _native.allclose
all = _native.all
any = _native.any
all_axis = _native.all_axis
any_axis = _native.any_axis
bitwise_and = _native.bitwise_and
bitwise_or = _native.bitwise_or
bitwise_xor = _native.bitwise_xor
bitwise_not = _native.bitwise_not
dot = _native.dot
squeeze = _native.squeeze
expand_dims = _native.expand_dims
concatenate = _native.concatenate
stack = _native.stack
ravel = _native.ravel
flatten = _native.flatten
clip = _native.clip
matmul = _native.matmul
neg = _native.neg
abs = _native.abs
sign = _native.sign
round = _native.round
isnan = _native.isnan
isinf = _native.isinf
isfinite = _native.isfinite
shape = _native.shape
ndim = _native.ndim
size = _native.size
dtype = _native.dtype
add = _native.add
subtract = _native.subtract
multiply = _native.multiply
divide = _native.divide
equal = _native.equal
not_equal = _native.not_equal
less = _native.less
less_equal = _native.less_equal
greater = _native.greater
greater_equal = _native.greater_equal
select = _native.select
count_true = _native.count_true
nonzero = _native.nonzero
argwhere = _native.argwhere
masked_fill = _native.masked_fill
sum = _native.sum
mean = _native.mean
min = _native.min
max = _native.max
argmin = _native.argmin
argmax = _native.argmax
argmin_axis = _native.argmin_axis
argmax_axis = _native.argmax_axis
cumsum = _native.cumsum
cumprod = _native.cumprod
cumsum_axis = _native.cumsum_axis
cumprod_axis = _native.cumprod_axis
nanmin = _native.nanmin
nanmax = _native.nanmax
nanmean = _native.nanmean
nanstd = _native.nanstd
sum_axis = _native.sum_axis
mean_axis = _native.mean_axis
min_axis = _native.min_axis
max_axis = _native.max_axis
transpose = _native.transpose
moveaxis = _native.moveaxis
swap_axes = _native.swap_axes
split = _native.split
repeat = _native.repeat
tile = _native.tile
flip = _native.flip
roll = _native.roll
sort = _native.sort
argsort = _native.argsort
argpartition = _native.argpartition
pad = _native.pad
searchsorted = _native.searchsorted
partition = _native.partition


def full(shape, fill_value=_UNSET, *, dtype=None, value=_UNSET):
    if fill_value is not _UNSET and value is not _UNSET:
        raise TypeError("full() received both 'fill_value' and 'value'")
    if fill_value is _UNSET:
        if value is _UNSET:
            raise TypeError("full() missing required argument: 'fill_value'")
        fill_value = value
    return _native.full(shape, fill_value, dtype)


def where(condition, x=_UNSET, y=_UNSET):
    condition = _array_like(condition)
    if x is _UNSET:
        if y is not _UNSET:
            raise TypeError("where() requires both x and y")
        return nonzero(condition)
    if y is _UNSET:
        raise TypeError("where() requires both x and y")
    return _native.where(condition, _optional_array_like(x), _optional_array_like(y))


def unique(
    value, *, axis=None, return_index=False, return_inverse=False, return_counts=False
):
    if axis is not None:
        raise _axis_error(axis, expected="axis=None")
    value = _array_like(value)
    result = _native.unique(value, return_index, return_inverse, return_counts)
    if not return_inverse:
        return result
    outputs = list(result)
    outputs[1 + return_index] = outputs[1 + return_index].reshape(value.shape)
    return tuple(outputs)


def take(value, indices, axis=None, *, mode="raise"):
    value = _array_like(value)
    if mode == "raise":
        return _native.take(value, indices, axis)
    if mode not in {"wrap", "clip"}:
        raise ValueError("mode must be 'raise', 'wrap', or 'clip'")
    indices = np.asarray(indices)
    if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
        raise TypeError("indices must be a one-dimensional integer sequence")
    if axis is None:
        length = value.size
    else:
        axis_index = integer_index(axis)
        normalized_axis = axis_index + value.ndim if axis_index < 0 else axis_index
        if normalized_axis < 0 or normalized_axis >= value.ndim:
            return _native.take(value, indices.tolist(), axis)
        length = value.shape[normalized_axis]
    if length == 0:
        return _native.take(value, indices.tolist(), axis)
    indices = indices % length if mode == "wrap" else np.clip(indices, 0, length - 1)
    return _native.take(value, indices.tolist(), axis)


def allclose(lhs, rhs, *, rtol=1e-5, atol=1e-8, equal_nan=False):
    return _native.allclose(lhs, rhs, rtol, atol, equal_nan)


def put(value, indices, values):
    value = _array_like(value)
    if _is_array_like(values):
        values = np.asarray(values, dtype=value.dtype)
    np.copyto(value, _native.put(value, indices, values))


def choose(indices, choices):
    indices = _array_like(indices)
    if indices.dtype.kind not in "iu":
        raise TypeError("choose indices must have an integer dtype")
    if not isinstance(choices, Sequence) or isinstance(
        choices, (str, bytes, bytearray)
    ):
        raise TypeError("choices must be a non-empty sequence")
    if not choices:
        raise ValueError("choices must be a non-empty sequence")
    if indices.size and (np.any(indices < 0) or np.any(indices >= len(choices))):
        raise ValueError("choose indices must be within the choices range")
    choices = [_array_or_scalar(choice) for choice in choices]
    dtype = np.result_type(*choices)
    result = np.asarray(choices[0], dtype=dtype)
    result = where(equal(indices, 0), result, result)
    for index, choice in enumerate(choices[1:], start=1):
        result = where(equal(indices, index), np.asarray(choice, dtype=dtype), result)
    return result


def broadcast_to(value, shape):
    try:
        return np.broadcast_to(_array_like(value), shape)
    except ValueError as error:
        raise _shape_error(error) from None


def broadcast_arrays(*values):
    values = [_array_or_scalar(value) for value in values]
    try:
        return np.broadcast_arrays(*values)
    except ValueError as error:
        raise _shape_error(error) from None


def copy(value):
    return np.array(_array_like(value), copy=True, order="C")


def zeros_like(value, *, dtype=None):
    value = _array_like(value)
    return zeros(value.shape, dtype=value.dtype if dtype is None else dtype)


def ones_like(value, *, dtype=None):
    value = _array_like(value)
    return ones(value.shape, dtype=value.dtype if dtype is None else dtype)


def empty_like(value, *, dtype=None):
    return np.empty_like(_array_like(value), dtype=dtype)


def copyto(destination, source, *, where=True):
    if not isinstance(destination, np.ndarray):
        raise TypeError("destination must be a NumPy ndarray")
    source = _array_like(source) if _is_array_like(source) else source
    if _is_array_like(where):
        where = _array_like(where)
        if where.dtype != np.dtype(bool):
            raise TypeError("where must have a boolean dtype")
    elif not isinstance(where, (bool, np.bool_)):
        raise TypeError("where must be a boolean scalar or array")
    try:
        np.copyto(destination, source, where=where)
    except ValueError as error:
        if "broadcast" in str(error):
            raise _shape_error(error) from None
        raise


def shares_memory(left, right, *, max_work=None):
    return bool(
        np.shares_memory(_array_like(left), _array_like(right), max_work=max_work)
    )


def may_share_memory(left, right, *, max_work=None):
    return bool(
        np.may_share_memory(_array_like(left), _array_like(right), max_work=max_work)
    )


def full_like(value, fill_value, *, dtype=None):
    value = _array_like(value)
    return full(value.shape, fill_value, dtype=value.dtype if dtype is None else dtype)


def ascontiguousarray(value, *, dtype=None):
    return np.ascontiguousarray(_array_like(value), dtype=dtype)


def _atleast(function, values):
    result = function(*[_array_or_scalar(value) for value in values])
    return result if len(values) == 1 else tuple(result)


def atleast_1d(*values):
    return _atleast(np.atleast_1d, values)


def atleast_2d(*values):
    return _atleast(np.atleast_2d, values)


def atleast_3d(*values):
    return _atleast(np.atleast_3d, values)


def _result_type_operand(value):
    if isinstance(value, (str, np.dtype)) or (
        isinstance(value, type) and issubclass(value, np.generic)
    ):
        return np.dtype(value)
    return _array_like(value) if _is_array_like(value) else value


def result_type(*values):
    return np.result_type(*[_result_type_operand(value) for value in values])


def promote_types(left, right):
    return np.promote_types(np.dtype(left), np.dtype(right))


def can_cast(source, target, *, casting="safe"):
    return bool(np.can_cast(source, target, casting=casting))


def issubdtype(dtype, kind):
    return bool(np.issubdtype(dtype, kind))


def min_scalar_type(value):
    dtype = np.min_scalar_type(value)
    if dtype == np.dtype(np.float16):
        return np.dtype(np.float32)
    if dtype.name not in _ATLAS_DTYPE_NAMES:
        raise TypeError(f"unsupported scalar dtype {dtype.name}")
    return dtype


def _atlas_dtype(dtype):
    if dtype is None:
        raise TypeError("dtype is required")
    try:
        dtype = np.dtype(dtype)
    except (TypeError, ValueError):
        raise _dtype_error(dtype) from None
    if dtype.name not in _ATLAS_DTYPE_NAMES:
        raise _dtype_error(dtype.name)
    return dtype


def finfo(dtype):
    dtype = _atlas_dtype(dtype)
    if dtype.kind != "f":
        raise TypeError("finfo requires a floating-point dtype")
    return np.finfo(dtype)


def iinfo(dtype):
    dtype = _atlas_dtype(dtype)
    if dtype.kind not in "iu":
        raise TypeError("iinfo requires an integer dtype")
    return np.iinfo(dtype)


def common_type(*values):
    return np.common_type(*[_array_or_scalar(value) for value in values])


def reshape(value, shape, *dimensions):
    value = _array_like(value)
    if dimensions:
        shape = (shape, *dimensions)
    elif isinstance(shape, np.ndarray):
        shape = shape.tolist()
    elif not isinstance(shape, Sequence) or isinstance(shape, (str, bytes, bytearray)):
        shape = (shape,)
    return _native.reshape(value, shape)


def linspace(start, stop, num, *, dtype=None, endpoint=True):
    return _native.linspace(start, stop, num, dtype, endpoint)


def astype(value, dtype, *, copy=True):
    return _native.astype(value, dtype, copy)


asarray = _coerce_arrays(asarray, required=((0, "value"),))
astype = _coerce_arrays(astype, required=((0, "value"),))

for _name in (
    "shape",
    "ndim",
    "size",
    "dtype",
    "empty_like",
    "count_true",
    "all",
    "any",
    "all_axis",
    "any_axis",
    "ravel",
    "flatten",
    "nonzero",
    "argwhere",
    "sum",
    "mean",
    "min",
    "max",
    "argmin",
    "argmax",
    "cumsum",
    "cumprod",
    "cumsum_axis",
    "cumprod_axis",
    "nanmin",
    "nanmax",
    "nanmean",
    "nanstd",
    "argmin_axis",
    "argmax_axis",
    "sum_axis",
    "mean_axis",
    "min_axis",
    "max_axis",
    "may_share_memory",
    "transpose",
    "moveaxis",
    "swap_axes",
    "split",
    "repeat",
    "tile",
    "flip",
    "roll",
    "sort",
    "argsort",
    "argpartition",
    "squeeze",
    "expand_dims",
    "partition",
    "put",
    "neg",
    "abs",
    "sign",
    "round",
    "isnan",
    "isinf",
    "isfinite",
):
    globals()[_name] = _coerce_arrays(globals()[_name], required=((0, "value"),))

for _name in (
    "add",
    "subtract",
    "multiply",
    "divide",
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
):
    globals()[_name] = _coerce_binary_operands(globals()[_name])

for _name in (
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "broadcast_arrays",
    "broadcast_to",
):
    globals()[_name] = _coerce_arrays(
        globals()[_name], required=((0, "lhs"),), optional=((1, "rhs"),)
    )

for _name in ("dot", "matmul", "allclose"):
    globals()[_name] = _coerce_arrays(
        globals()[_name], required=((0, "lhs"), (1, "rhs"))
    )

for _name in ("select", "masked_fill"):
    globals()[_name] = _coerce_arrays(
        globals()[_name], required=((0, "value"), (1, "mask"))
    )

bitwise_not = _coerce_arrays(bitwise_not, required=((0, "value"),))
pad = _coerce_arrays(pad, required=((0, "array"),))
searchsorted = _coerce_searchsorted(searchsorted)
clip = _coerce_arrays(clip, required=((0, "value"),))
concatenate = _coerce_array_collection(concatenate)
stack = _coerce_array_collection(stack)
