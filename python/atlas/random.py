"""Seeded random-number generation and probability distribution sampling."""

from . import _random as _bindings
from ._exports import RANDOM_EXPORTS
from ._support import _document

__all__ = tuple(RANDOM_EXPORTS)

globals().update(
    {
        name: _document(
            getattr(_bindings, name),
            f"Create or configure the Atlas random generator with {name.replace('_', ' ')}.",
        )
        if callable(getattr(_bindings, name))
        and not isinstance(getattr(_bindings, name), type)
        else getattr(_bindings, name)
        for name in __all__
    }
)
