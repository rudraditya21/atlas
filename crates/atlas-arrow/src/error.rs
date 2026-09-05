use atlas_ndarray::AtlasNdError;
use thiserror::Error;

pub type AtlasArrowResult<T> = Result<T, AtlasArrowError>;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AtlasArrowError {
    #[error("invalid input rank for {op}: expected {expected}, got rank {rank}")]
    InvalidInputRank { op: &'static str, expected: &'static str, rank: usize },

    #[error("null values are not supported for {op}")]
    NullValues { op: &'static str },

    #[error("column name count mismatch for {op}: expected {expected}, got {actual}")]
    ColumnNameCountMismatch { op: &'static str, expected: usize, actual: usize },

    #[error("failed to create record batch: {reason}")]
    RecordBatch { reason: String },

    #[error(transparent)]
    NdArray(#[from] AtlasNdError),
}
