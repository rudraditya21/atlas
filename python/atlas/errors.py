"""Public exceptions raised by Atlas."""


class AtlasError(Exception):
    """Base exception for Atlas failures."""


class ShapeError(AtlasError):
    """Raised when operand shapes are incompatible."""


class AxisError(ShapeError):
    """Raised when an axis is invalid for an operand."""


class SliceError(ShapeError):
    """Raised when a slice is invalid for an operand."""


class NumericError(AtlasError):
    """Raised when a numeric input or computation is invalid."""


class ModelError(AtlasError):
    """Raised when a machine-learning model operation is invalid."""


__all__ = [
    "AtlasError",
    "AxisError",
    "ModelError",
    "NumericError",
    "ShapeError",
    "SliceError",
]
