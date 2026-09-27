"""Random-number generation."""

from . import _random as _bindings
from ._exports import RANDOM_EXPORTS

__all__ = RANDOM_EXPORTS

globals().update({name: getattr(_bindings, name) for name in __all__})
