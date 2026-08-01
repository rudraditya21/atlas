use thiserror::Error;

use atlas_ndarray::AtlasNdError;

pub type AtlasLinalgResult<T> = Result<T, AtlasLinalgError>;

#[derive(Debug, Error)]
pub enum AtlasLinalgError {
    #[error("invalid operand rank for {op}: left rank {left}, right rank {right}")]
    InvalidOperandRank {
        op: &'static str,
        left: usize,
        right: usize,
    },

    #[error(
        "shape mismatch for {op}: left shape {left:?}, right shape {right:?}: {reason}"
    )]
    ShapeMismatch {
        op: &'static str,
        left: Vec<usize>,
        right: Vec<usize>,
        reason: &'static str,
    },

    #[error(transparent)]
    NdArray(#[from] AtlasNdError),
}
