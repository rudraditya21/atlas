use thiserror::Error;

pub type AtlasArrowResult<T> = Result<T, AtlasArrowError>;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AtlasArrowError {
    #[error("invalid input rank for {op}: expected rank-1 vector, got rank {rank}")]
    InvalidInputRank { op: &'static str, rank: usize },
}
