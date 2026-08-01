use atlas_ndarray::AtlasNdError;
use thiserror::Error;

pub type AtlasStatsResult<T> = Result<T, AtlasStatsError>;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AtlasStatsError {
    #[error("invalid input rank for {op}: expected {expected}, got rank {rank}")]
    InvalidInputRank { op: &'static str, expected: &'static str, rank: usize },

    #[error("shape mismatch for {op}: left {left:?}, right {right:?}: {reason}")]
    ShapeMismatch { op: &'static str, left: Vec<usize>, right: Vec<usize>, reason: &'static str },

    #[error("empty input for {op}")]
    EmptyInput { op: &'static str },

    #[error("numeric conversion failed for {op}")]
    NumericConversionFailed { op: &'static str },

    #[error("zero variance for {op}")]
    ZeroVariance { op: &'static str },

    #[error(transparent)]
    NdArray(#[from] AtlasNdError),
}

#[cfg(test)]
mod tests {
    use super::AtlasStatsError;

    #[test]
    fn error_messages_follow_consistent_style() {
        assert_eq!(
            AtlasStatsError::ShapeMismatch {
                op: "correlation",
                left: vec![3],
                right: vec![2],
                reason: "vector lengths must match",
            }
            .to_string(),
            "shape mismatch for correlation: left [3], right [2]: vector lengths must match"
        );
        assert_eq!(
            AtlasStatsError::ZeroVariance { op: "correlation" }.to_string(),
            "zero variance for correlation"
        );
    }
}
