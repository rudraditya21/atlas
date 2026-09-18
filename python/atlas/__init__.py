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

__all__ = [
    "__version__",
    "AtlasError",
    "AxisError",
    "ModelError",
    "NumericError",
    "ShapeError",
    "SliceError",
]
