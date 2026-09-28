"""Machine-learning models, preprocessing, metrics, and split utilities."""

from . import _ml as _bindings
from . import _models
from ._exports import ML_EXPORTS
from ._support import _document

__all__ = tuple(sorted((*_models.__all__, *ML_EXPORTS)))

globals().update({name: getattr(_models, name) for name in _models.__all__})
globals().update(
    {
        name: _document(
            getattr(_bindings, name),
            f"Run the Atlas machine learning operation {name.replace('_', ' ')}.",
        )
        for name in ML_EXPORTS
    }
)
