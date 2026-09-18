"""Python bindings for Atlas."""

from . import _native

__version__ = _native.version()

__all__ = ["__version__"]
