mod core;
mod dense;
mod factorization;
pub(crate) mod internal;

pub use core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand};
pub use dense::{dot, matmul};
pub use factorization::{
    CholeskyFactorization, LuFactorization, QrFactorization, cholesky, lu, qr,
};

pub type LUFactorization<T> = LuFactorization<T>;
pub type QRFactorization<T> = QrFactorization<T>;
