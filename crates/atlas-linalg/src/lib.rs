mod core;
mod dense;
mod factorization;
pub(crate) mod internal;

pub use core::error::{AtlasLinalgError, AtlasLinalgResult};
pub use core::operand::LinalgOperand;
pub use dense::dot::{DotOutput, dot};
pub use dense::matmul::matmul;
pub use dense::norm::norm;
pub use factorization::cholesky::{CholeskyFactorization, cholesky};
pub use factorization::lu::{LuFactorization, lu};
pub use factorization::qr::{QrFactorization, qr};

pub type LUFactorization<T> = LuFactorization<T>;
pub type QRFactorization<T> = QrFactorization<T>;
