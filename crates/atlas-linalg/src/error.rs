use thiserror::Error;

use atlas_ndarray::AtlasNdError;

pub type AtlasLinalgResult<T> = Result<T, AtlasLinalgError>;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AtlasLinalgError {
    #[error("invalid operand rank for {op}: left rank {left}, right rank {right}")]
    InvalidOperandRank { op: &'static str, left: usize, right: usize },

    #[error("shape mismatch for {op}: left shape {left:?}, right shape {right:?}: {reason}")]
    ShapeMismatch { op: &'static str, left: Vec<usize>, right: Vec<usize>, reason: &'static str },

    #[error("invalid input rank for {op}: expected {expected}, got rank {rank}")]
    InvalidInputRank { op: &'static str, expected: &'static str, rank: usize },

    #[error("invalid input for {op} with shape {shape:?}: {reason}")]
    InvalidInputShape { op: &'static str, shape: Vec<usize>, reason: &'static str },

    #[error("singular matrix encountered during {op} at pivot {pivot}")]
    SingularMatrix { op: &'static str, pivot: usize },

    #[error("rank deficient matrix encountered during {op} at column {column}")]
    RankDeficientMatrix { op: &'static str, column: usize },

    #[error("matrix is not positive definite during {op} at diagonal {index}")]
    NotPositiveDefinite { op: &'static str, index: usize },

    #[error(transparent)]
    NdArray(#[from] AtlasNdError),
}
