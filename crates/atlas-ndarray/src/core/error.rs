use crate::{CastMode, DType};
use thiserror::Error;

pub type AtlasNdResult<T> = Result<T, AtlasNdError>;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AtlasNdError {
    #[error("shape mismatch: expected {expected} elements, got {actual}")]
    ShapeMismatch { expected: usize, actual: usize },

    #[error("shape overflow for {op}: {shape:?}")]
    ShapeOverflow { op: &'static str, shape: Vec<usize> },

    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("invalid axis: axis {axis} is out of bounds for ndim {ndim}")]
    InvalidAxis { axis: i64, ndim: usize },

    #[error("index out of bounds on axis {axis}: index {index}, length {dim}")]
    IndexOutOfBounds { axis: usize, index: i64, dim: usize },

    #[error("invalid slice on axis {axis}: start {start}, len {len}, dim {dim}")]
    InvalidSlice { axis: usize, start: usize, len: usize, dim: usize },

    #[error("invalid reshape from {from:?} to {to:?}: {reason}")]
    InvalidReshape { from: Vec<usize>, to: Vec<usize>, reason: &'static str },

    #[error(
        "invalid broadcast between {lhs:?} and {rhs:?}: axis {axis} has incompatible dimensions {lhs_dim} and {rhs_dim}"
    )]
    InvalidBroadcast {
        lhs: Vec<usize>,
        rhs: Vec<usize>,
        axis: usize,
        lhs_dim: usize,
        rhs_dim: usize,
    },

    #[error("empty input for {op}")]
    EmptyReduction { op: &'static str },

    #[error("numeric conversion failed for {op}")]
    NumericConversionFailed { op: &'static str },

    #[error("division by zero for {op}")]
    DivisionByZero { op: &'static str },

    #[error("invalid argument for {op}: {reason}")]
    InvalidArgument { op: &'static str, reason: &'static str },

    #[error("invalid cast from {from} to {to} under {mode:?} mode")]
    InvalidCast { from: DType, to: DType, mode: CastMode },

    #[error("invalid shape")]
    InvalidShape,
}

#[cfg(test)]
mod tests {
    use super::AtlasNdError;
    use crate::{CastMode, DType};

    #[test]
    fn error_messages_follow_consistent_style() {
        assert_eq!(
            AtlasNdError::ShapeMismatch { expected: 4, actual: 3 }.to_string(),
            "shape mismatch: expected 4 elements, got 3"
        );
        assert_eq!(
            AtlasNdError::ShapeOverflow { op: "element count", shape: vec![usize::MAX, 2] }
                .to_string(),
            format!("shape overflow for element count: {:?}", vec![usize::MAX, 2])
        );
        assert_eq!(AtlasNdError::EmptyReduction { op: "mean" }.to_string(), "empty input for mean");
        assert_eq!(
            AtlasNdError::NumericConversionFailed { op: "mean" }.to_string(),
            "numeric conversion failed for mean"
        );
        assert_eq!(
            AtlasNdError::DivisionByZero { op: "division" }.to_string(),
            "division by zero for division"
        );
        assert_eq!(
            AtlasNdError::InvalidCast { from: DType::I64, to: DType::I32, mode: CastMode::Checked }
                .to_string(),
            "invalid cast from int64 to int32 under Checked mode"
        );
    }
}
