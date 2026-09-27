"""Atlas numerical computing, machine learning, and Arrow interoperability APIs."""

from . import interop, linalg, ml, random, statistics
from ._public import *

# Common NumPy-style functions are deliberately available without a submodule import.
from ._public import (
    arange,
    asarray,
    concatenate,
    dot,
    full,
    linspace,
    matmul,
    max,
    mean,
    min,
    ones,
    reshape,
    sort,
    stack,
    sum,
    take,
    transpose,
    unique,
    where,
    zeros,
)

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
