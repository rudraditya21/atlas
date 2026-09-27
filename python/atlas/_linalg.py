"""Linear algebra bindings exposed by Atlas."""

from . import _native
from ._support import _coerce_arrays

ConjugateGradientResult = _native.ConjugateGradientResult
norm = _native.norm
trace = _native.trace
diag = _native.diag
matrix_norm = _native.matrix_norm
det = _native.det
inverse = _native.inverse
solve = _native.solve
solve_transpose = _native.solve_transpose
slogdet = _native.slogdet
cholesky = _native.cholesky
solve_spd = _native.solve_spd
qr = _native.qr
least_squares = _native.least_squares
matrix_rank = _native.matrix_rank
symmetric_eigendecomposition = _native.symmetric_eigendecomposition
conjugate_gradient = _native.conjugate_gradient

for _name in (
    "norm",
    "trace",
    "diag",
    "matrix_norm",
    "qr",
    "matrix_rank",
    "symmetric_eigendecomposition",
    "det",
    "slogdet",
    "inverse",
    "cholesky",
):
    globals()[_name] = _coerce_arrays(globals()[_name], required=((0, "value"),))

solve = _coerce_arrays(solve, required=((0, "matrix"), (1, "rhs")))
solve_transpose = _coerce_arrays(solve_transpose, required=((0, "matrix"), (1, "rhs")))
solve_spd = _coerce_arrays(solve_spd, required=((0, "matrix"), (1, "rhs")))
least_squares = _coerce_arrays(least_squares, required=((0, "matrix"), (1, "rhs")))
conjugate_gradient = _coerce_arrays(
    conjugate_gradient, required=((0, "matrix"), (1, "rhs"))
)
