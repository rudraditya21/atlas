"""Linear algebra decompositions, solvers, norms, and result types."""

from . import _linalg as _bindings
from ._exports import LINALG_EXPORTS

__all__ = tuple(LINALG_EXPORTS)

globals().update({name: getattr(_bindings, name) for name in __all__})
