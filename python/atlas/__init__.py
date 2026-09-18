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

__all__ = [
    "__version__",
    "arange",
    "asarray",
    "AtlasError",
    "AxisError",
    "ModelError",
    "NumericError",
    "ShapeError",
    "SliceError",
    "full",
    "dtype",
    "ndim",
    "ones",
    "shape",
    "size",
    "zeros",
]
