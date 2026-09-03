mod core;
mod dense;
mod factorization;
pub(crate) mod internal;

pub use core::error::{AtlasLinalgError, AtlasLinalgResult};
pub use core::operand::LinalgOperand;
pub use dense::diag::{diag, diag_view};
pub use dense::dot::{DotOutput, dot};
pub use dense::matmul::matmul;
pub use dense::norm::norm;
pub use dense::trace::trace;
pub use dense::triangular::{solve_lower_triangular, solve_upper_triangular};
pub use factorization::cholesky::{CholeskyFactorization, cholesky, solve_spd};
pub use factorization::lu::{LuFactorization, det, lu, slogdet, solve};
pub use factorization::qr::{QrFactorization, least_squares, qr};

pub type LUFactorization<T> = LuFactorization<T>;
pub type QRFactorization<T> = QrFactorization<T>;
