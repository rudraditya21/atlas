"""Descriptive, weighted, pairwise, and axis-aware statistical functions."""

from . import _statistics as _bindings
from ._exports import STATISTICS_EXPORTS
from ._support import _document

__all__ = tuple(STATISTICS_EXPORTS)

globals().update(
    {
        name: _document(
            getattr(_bindings, name),
            f"Compute the Atlas statistic {name.replace('_', ' ')}.",
        )
        for name in __all__
    }
)
