"""Descriptive, weighted, pairwise, and axis-aware statistical functions."""

from . import _statistics as _bindings
from ._exports import STATISTICS_EXPORTS

__all__ = tuple(STATISTICS_EXPORTS)

globals().update({name: getattr(_bindings, name) for name in __all__})
