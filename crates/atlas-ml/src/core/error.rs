use atlas_ndarray::AtlasNdError;
use thiserror::Error;

pub type AtlasMlResult<T> = Result<T, AtlasMlError>;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AtlasMlError {
    #[error("invalid input rank for {op}: expected {expected}, got rank {rank}")]
    InvalidInputRank { op: &'static str, expected: &'static str, rank: usize },

    #[error("shape mismatch for {op}: left {left:?}, right {right:?}: {reason}")]
    ShapeMismatch { op: &'static str, left: Vec<usize>, right: Vec<usize>, reason: &'static str },

    #[error("empty input for {op}")]
    EmptyInput { op: &'static str },

    #[error("non-finite input for {op}")]
    NonFiniteInput { op: &'static str },

    #[error("invalid argument for {op}: {reason}")]
    InvalidArgument { op: &'static str, reason: &'static str },

    #[error(transparent)]
    NdArray(#[from] AtlasNdError),
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::AtlasNdError;

    use super::AtlasMlError;

    #[test]
    fn error_messages_follow_consistent_style() {
        assert_eq!(
            AtlasMlError::InvalidInputRank {
                op: "knn_fit",
                expected: "a rank-2 feature matrix",
                rank: 1,
            }
            .to_string(),
            "invalid input rank for knn_fit: expected a rank-2 feature matrix, got rank 1"
        );
        assert_eq!(
            AtlasMlError::ShapeMismatch {
                op: "knn_fit",
                left: vec![3, 2],
                right: vec![2],
                reason: "sample counts must match",
            }
            .to_string(),
            "shape mismatch for knn_fit: left [3, 2], right [2]: sample counts must match"
        );
        assert_eq!(
            AtlasMlError::EmptyInput { op: "knn_fit" }.to_string(),
            "empty input for knn_fit"
        );
        assert_eq!(
            AtlasMlError::NonFiniteInput { op: "knn_fit" }.to_string(),
            "non-finite input for knn_fit"
        );
        assert_eq!(
            AtlasMlError::InvalidArgument { op: "knn_fit", reason: "k must be positive" }
                .to_string(),
            "invalid argument for knn_fit: k must be positive"
        );
        assert_eq!(AtlasMlError::NdArray(AtlasNdError::InvalidShape).to_string(), "invalid shape");
    }
}
