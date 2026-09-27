"""Public Python API for Atlas."""

from ._public import *

# Root-only compatibility helpers and NumPy-compatible aliases remain intentionally explicit.
from ._public import (
    __all__,
    atleast_1d,
    atleast_2d,
    atleast_3d,
    broadcast_arrays,
    broadcast_to,
    empty_like,
    may_share_memory,
    put,
    std,
    swapaxes,
    var,
)
