"""NumPy and Apache Arrow primitive-array and record-batch interoperability."""

from . import _interop as _bindings
from ._exports import INTEROP_EXPORTS
from ._support import _document

__all__ = tuple(INTEROP_EXPORTS)

globals().update(
    {
        name: _document(
            getattr(_bindings, name),
            f"Convert data with the Atlas Arrow interoperability operation {name.replace('_', ' ')}.",
        )
        for name in __all__
    }
)
