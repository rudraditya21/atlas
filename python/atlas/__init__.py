"""Python bindings for Atlas."""

from collections.abc import Sequence
from functools import wraps

import numpy as np

from . import _native
from .errors import (
    AtlasError,
    AxisError,
    ModelError,
    NumericError,
    ShapeError,
    SliceError,
)

__version__ = _native.version()


def _array_like(value):
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return np.asarray(value)
    raise TypeError("expected a NumPy ndarray or Python sequence")


def _is_array_like(value):
    return isinstance(value, np.ndarray) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    )


def _optional_array_like(value):
    if _is_array_like(value):
        return _array_like(value)
    return value


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
    """Coerce binary operands to their NumPy-promoted dtype."""

    @wraps(function)
    def wrapper(lhs, rhs):
        lhs_value = _array_like(lhs) if _is_array_like(lhs) else lhs
        rhs_value = _array_like(rhs) if _is_array_like(rhs) else rhs
        dtype = np.result_type(lhs_value, rhs_value)
        return function(
            np.asarray(lhs_value, dtype=dtype), np.asarray(rhs_value, dtype=dtype)
        )

    return wrapper


def _coerce_array_collection(function):
    @wraps(function)
    def wrapper(arrays, axis=0):
        return function([_array_like(value) for value in arrays], axis)

    return wrapper


def _with_keepdims(function):
    @wraps(function)
    def wrapper(value, axis=None, keepdims=False):
        if keepdims and axis is None:
            raise ValueError("keepdims requires a single axis")
        result = function(value, axis=axis)
        return np.expand_dims(result, axis) if keepdims else result

    return wrapper


asarray = _native.asarray
zeros = _native.zeros
ones = _native.ones
eye = _native.eye
identity = _native.identity
full = _native.full
arange = _native.arange
linspace = _native.linspace
astype = _native.astype
allclose = _native.allclose
all = _native.all
any = _native.any
all_axis = _native.all_axis
any_axis = _native.any_axis
where = _native.where
bitwise_and = _native.bitwise_and
bitwise_or = _native.bitwise_or
bitwise_xor = _native.bitwise_xor
bitwise_not = _native.bitwise_not
take = _native.take
dot = _native.dot
norm = _native.norm
trace = _native.trace
squeeze = _native.squeeze
expand_dims = _native.expand_dims
concatenate = _native.concatenate
stack = _native.stack
diag = _native.diag
matrix_norm = _native.matrix_norm
det = _native.det
inverse = _native.inverse
solve = _native.solve
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
variance = _native.variance
stddev = _native.stddev
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
reshape = _native.reshape
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
unique = _native.unique
pad = _native.pad
searchsorted = _native.searchsorted
partition = _native.partition


_UNSET = object()


def full(shape, fill_value=_UNSET, dtype=None, *, value=_UNSET):
    if fill_value is not _UNSET and value is not _UNSET:
        raise TypeError("full() received both 'fill_value' and 'value'")
    if fill_value is _UNSET:
        if value is _UNSET:
            raise TypeError("full() missing required argument: 'fill_value'")
        fill_value = value
    return _native.full(shape, fill_value, dtype)


asarray = _coerce_arrays(asarray, required=((0, "value"),))
astype = _coerce_arrays(astype, required=((0, "value"),))

for _name in (
    "shape",
    "ndim",
    "size",
    "dtype",
    "count_true",
    "all",
    "any",
    "all_axis",
    "any_axis",
    "take",
    "norm",
    "trace",
    "diag",
    "matrix_norm",
    "det",
    "inverse",
    "ravel",
    "flatten",
    "nonzero",
    "argwhere",
    "sum",
    "mean",
    "min",
    "max",
    "variance",
    "stddev",
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
    "reshape",
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
    "unique",
    "squeeze",
    "expand_dims",
    "partition",
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
):
    globals()[_name] = _coerce_arrays(
        globals()[_name], required=((0, "lhs"),), optional=((1, "rhs"),)
    )

for _name in ("dot", "matmul", "allclose"):
    globals()[_name] = _coerce_arrays(
        globals()[_name], required=((0, "lhs"), (1, "rhs"))
    )

solve = _coerce_arrays(solve, required=((0, "matrix"), (1, "rhs")))

for _name in ("select", "masked_fill"):
    globals()[_name] = _coerce_arrays(
        globals()[_name], required=((0, "value"), (1, "mask"))
    )

where = _coerce_arrays(
    where, required=((0, "condition"),), optional=((1, "x"), (2, "y"))
)
bitwise_not = _coerce_arrays(bitwise_not, required=((0, "value"),))
pad = _coerce_arrays(pad, required=((0, "array"),))
searchsorted = _coerce_arrays(
    searchsorted, required=((0, "sorted"),), optional=((1, "values"),)
)
clip = _coerce_arrays(clip, required=((0, "value"),))
concatenate = _coerce_array_collection(concatenate)
stack = _coerce_array_collection(stack)
swapaxes = swap_axes
var = variance
std = stddev

for _name in ("all", "any", "sum", "mean", "min", "max", "argmin", "argmax"):
    globals()[_name] = _with_keepdims(globals()[_name])

__all__ = [
    "AtlasError",
    "AxisError",
    "ModelError",
    "NumericError",
    "ShapeError",
    "SliceError",
    "__version__",
    "add",
    "abs",
    "allclose",
    "all",
    "all_axis",
    "argmax",
    "argmax_axis",
    "argmin",
    "argmin_axis",
    "any",
    "any_axis",
    "argwhere",
    "argpartition",
    "argsort",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "arange",
    "linspace",
    "astype",
    "asarray",
    "count_true",
    "cumprod",
    "cumprod_axis",
    "cumsum",
    "cumsum_axis",
    "clip",
    "concatenate",
    "divide",
    "diag",
    "det",
    "dot",
    "dtype",
    "eye",
    "identity",
    "equal",
    "expand_dims",
    "flatten",
    "flip",
    "full",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "masked_fill",
    "matmul",
    "matrix_norm",
    "max",
    "max_axis",
    "mean",
    "mean_axis",
    "min",
    "min_axis",
    "nanmax",
    "nanmean",
    "nanmin",
    "nanstd",
    "norm",
    "multiply",
    "moveaxis",
    "neg",
    "ndim",
    "nonzero",
    "not_equal",
    "ones",
    "pad",
    "partition",
    "select",
    "searchsorted",
    "shape",
    "size",
    "solve",
    "split",
    "sort",
    "subtract",
    "sum",
    "sum_axis",
    "swap_axes",
    "swapaxes",
    "squeeze",
    "stddev",
    "std",
    "stack",
    "take",
    "tile",
    "reshape",
    "ravel",
    "repeat",
    "roll",
    "round",
    "sign",
    "isnan",
    "inverse",
    "isinf",
    "isfinite",
    "transpose",
    "trace",
    "unique",
    "variance",
    "var",
    "where",
    "zeros",
]
