"""Public API assembly for :mod:`atlas`."""

from functools import wraps

import numpy as np

from . import _arrays, _interop, _linalg, _ml, _native, _random, _statistics
from . import _errors, _models
from ._exports import (
    ARRAY_EXPORTS,
    NUMPY_COMPATIBILITY_ALIASES,
    INTEROP_EXPORTS,
    LINALG_EXPORTS,
    ML_EXPORTS,
    RANDOM_EXPORTS,
    STATISTICS_EXPORTS,
)

__version__ = _native.version()


def show_config() -> dict[str, object]:
    """Return the active numerical backend configuration."""

    return _native._backend_config()


def _with_keepdims(function):
    def wrapper(value, axis=None, *, keepdims=False):
        if keepdims and axis is None:
            raise ValueError("keepdims requires a single axis")
        result = function(value, axis=axis)
        return np.expand_dims(result, axis) if keepdims else result

    wrapper.__name__ = function.__name__
    wrapper.__doc__ = function.__doc__
    return wrapper


def _publish(function):
    """Expose a native callable as a public Atlas function."""

    @wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    wrapper.__module__ = "atlas"
    return wrapper


_DOMAIN_EXPORTS = (
    (_arrays, ARRAY_EXPORTS),
    (_linalg, LINALG_EXPORTS),
    (_statistics, STATISTICS_EXPORTS),
    (_random, RANDOM_EXPORTS),
    (_ml, ML_EXPORTS),
    (_interop, INTEROP_EXPORTS),
    (_errors, _errors.__all__),
    (_models, _models.__all__),
)

__all__ = sorted(
    (
        "__version__",
        "show_config",
        *NUMPY_COMPATIBILITY_ALIASES,
        *(name for _, names in _DOMAIN_EXPORTS for name in names),
    )
)

for _module, _names in _DOMAIN_EXPORTS:
    for _name in _names:
        globals()[_name] = getattr(_module, _name)

for _name in (
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "broadcast_arrays",
    "broadcast_to",
    "empty_like",
    "may_share_memory",
    "put",
):
    _value = getattr(_arrays, _name)
    _value.__module__ = "atlas"
    globals()[_name] = _value

for _name in ("all", "any", "sum", "mean", "min", "max", "argmin", "argmax"):
    globals()[_name] = _with_keepdims(globals()[_name])

for _name in __all__:
    if _name in {"std", "swapaxes", "var"}:
        continue
    _value = globals()[_name]
    if isinstance(_value, type):
        _value.__module__ = "atlas"
    elif callable(_value):
        if getattr(_value, "__module__", None) == _native.__name__:
            globals()[_name] = _publish(_value)
        else:
            _value.__module__ = "atlas"
        if not globals()[_name].__doc__:
            globals()[
                _name
            ].__doc__ = f"Run the Atlas public operation {_name.replace('_', ' ')}."

# Keep familiar NumPy spellings at the package root while using Atlas canonical names internally.
swapaxes = swap_axes
var = variance
std = stddev
