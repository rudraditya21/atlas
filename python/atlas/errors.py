"""Public exceptions raised by Atlas."""


class AtlasError(Exception):
    """Base exception for Atlas failures."""


class ShapeError(AtlasError, ValueError):
    """Raised as a ``ValueError`` when operand shapes are incompatible."""


class AxisError(ShapeError, IndexError):
    """Raised as an ``IndexError`` when an axis is invalid for an operand."""


class SliceError(ShapeError, IndexError):
    """Raised as an ``IndexError`` when a slice is invalid for an operand."""


class NumericError(AtlasError, ValueError):
    """Raised as a ``ValueError`` when a numeric input or computation is invalid."""


class ModelError(AtlasError, ValueError):
    """Raised as a ``ValueError`` when a machine-learning model operation is invalid."""


__all__ = [
    "AtlasError",
    "AxisError",
    "ModelError",
    "NumericError",
    "ShapeError",
    "SliceError",
]
