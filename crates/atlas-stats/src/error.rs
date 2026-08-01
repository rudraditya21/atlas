use atlas_ndarray::AtlasNdError;
use thiserror::Error;

pub type AtlasStatsResult<T> = Result<T, AtlasStatsError>;

#[derive(Debug, Error)]
pub enum AtlasStatsError {
    #[error("invalid input rank for {op}: expected {expected}, got rank {rank}")]
    InvalidInputRank { op: &'static str, expected: &'static str, rank: usize },

    #[error("shape mismatch for {op}: left shape {left:?}, right shape {right:?}: {reason}")]
    ShapeMismatch { op: &'static str, left: Vec<usize>, right: Vec<usize>, reason: &'static str },

    #[error("empty input for {op}")]
    EmptyInput { op: &'static str },

    #[error("numeric conversion failed during {op}")]
    NumericConversionFailed { op: &'static str },

    #[error("zero variance encountered during {op}")]
    ZeroVariance { op: &'static str },

    #[error(transparent)]
    NdArray(#[from] AtlasNdError),
}
