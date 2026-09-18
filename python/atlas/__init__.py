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
    "ones",
    "zeros",
]
