mod core;
mod dense;
mod factorization;
pub(crate) mod internal;

pub use core::{
    error::{AtlasLinalgError, AtlasLinalgResult},
    operand::LinalgOperand,
};

pub use dense::{
    condition::condition_number,
    diag::{diag, diag_view},
    dot::{DotOutput, batched_dot, dot},
    matmul::matmul,
    norm::{MatrixNorm, matrix_norm, norm},
    trace::trace,
    triangular::{solve_lower_triangular, solve_upper_triangular},
};
pub use factorization::{
    cholesky::{CholeskyFactorization, cholesky, solve_spd},
    lu::{LuFactorization, det, inverse, lu, slogdet, solve, solve_transpose},
    qr::{QrFactorization, least_squares, matrix_rank, qr},
};

pub type LUFactorization<T> = LuFactorization<T>;
pub type QRFactorization<T> = QrFactorization<T>;
