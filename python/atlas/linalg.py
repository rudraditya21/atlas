"""Linear algebra decompositions, solvers, norms, and result types."""

from . import _linalg as _bindings
from ._exports import LINALG_EXPORTS
from ._support import _document

__all__ = tuple(LINALG_EXPORTS)

globals().update(
    {
        name: _document(
            getattr(_bindings, name),
            f"Perform the Atlas linear algebra operation {name.replace('_', ' ')}.",
        )
        if callable(getattr(_bindings, name))
        and not isinstance(getattr(_bindings, name), type)
        else getattr(_bindings, name)
        for name in __all__
    }
)
