use thiserror::Error;

pub type AtlasNdResult<T> = Result<T, AtlasNdError>;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AtlasNdError {
    #[error("shape does not match data length: expected {expected} elements, got {actual}")]
    ShapeMismatch { expected: usize, actual: usize },

    #[error("dimension mismatch: expected {expected} dimensions, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("index {index} is out of bounds for axis {axis} with length {dim}")]
    IndexOutOfBounds {
        axis: usize,
        index: usize,
        dim: usize,
    },

    #[error("invalid shape")]
    InvalidShape,
}
