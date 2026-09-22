"""Python bindings for Atlas."""

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
asarray = _native.asarray
zeros = _native.zeros
ones = _native.ones
full = _native.full
arange = _native.arange
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
swap_axes = _native.swap_axes
split = _native.split
repeat = _native.repeat
tile = _native.tile
flip = _native.flip
roll = _native.roll

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
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "arange",
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
    "neg",
    "ndim",
    "nonzero",
    "not_equal",
    "ones",
    "select",
    "shape",
    "size",
    "solve",
    "split",
    "subtract",
    "sum",
    "sum_axis",
    "swap_axes",
    "squeeze",
    "stddev",
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
    "variance",
    "where",
    "zeros",
]
