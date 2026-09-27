"""Arrow interoperability functions."""

from . import _interop as _bindings
from ._exports import INTEROP_EXPORTS

__all__ = INTEROP_EXPORTS

globals().update({name: getattr(_bindings, name) for name in __all__})
