mod dense;
mod error;
mod factorization;
mod operand;

pub use dense::{dot, matmul};
pub use error::{AtlasLinalgError, AtlasLinalgResult};
pub use factorization::{
    CholeskyFactorization, LuFactorization, QrFactorization, cholesky, lu, qr,
};
pub use operand::LinalgOperand;

pub type LUFactorization<T> = LuFactorization<T>;
pub type QRFactorization<T> = QrFactorization<T>;
