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
shape = _native.shape
ndim = _native.ndim
size = _native.size
dtype = _native.dtype
add = _native.add
subtract = _native.subtract
multiply = _native.multiply
divide = _native.divide

__all__ = [
    "__version__",
    "arange",
    "add",
    "asarray",
    "AtlasError",
    "AxisError",
    "ModelError",
    "NumericError",
    "ShapeError",
    "SliceError",
    "full",
    "dtype",
    "divide",
    "ndim",
    "multiply",
    "ones",
    "shape",
    "size",
    "subtract",
    "zeros",
]
