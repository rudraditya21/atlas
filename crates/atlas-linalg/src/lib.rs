pub mod dense;
pub mod error;
pub mod factorization;
pub mod operand;

pub use dense::{dot, matmul};
pub use error::{AtlasLinalgError, AtlasLinalgResult};
pub use factorization::{
    CholeskyFactorization, LuFactorization, QrFactorization, cholesky, lu, qr,
};
pub use operand::LinalgOperand;
