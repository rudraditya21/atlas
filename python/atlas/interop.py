"""NumPy and Apache Arrow primitive-array and record-batch interoperability."""

from . import _interop as _bindings
from ._exports import INTEROP_EXPORTS

__all__ = tuple(INTEROP_EXPORTS)

globals().update({name: getattr(_bindings, name) for name in __all__})
