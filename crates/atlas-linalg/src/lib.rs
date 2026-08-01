pub mod dense;
pub mod error;

pub use dense::{dot, matmul};
pub use error::{AtlasLinalgError, AtlasLinalgResult};
