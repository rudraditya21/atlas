pub mod dense;
pub mod error;
pub mod operand;

pub use dense::{dot, matmul};
pub use error::{AtlasLinalgError, AtlasLinalgResult};
pub use operand::LinalgOperand;
