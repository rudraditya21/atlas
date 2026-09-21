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
masked_fill = _native.masked_fill

__all__ = [
    "AtlasError",
    "AxisError",
    "ModelError",
    "NumericError",
    "ShapeError",
    "SliceError",
    "__version__",
    "add",
    "arange",
    "astype",
    "asarray",
    "count_true",
    "divide",
    "dtype",
    "equal",
    "full",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "masked_fill",
    "multiply",
    "ndim",
    "nonzero",
    "not_equal",
    "ones",
    "select",
    "shape",
    "size",
    "subtract",
    "zeros",
]
