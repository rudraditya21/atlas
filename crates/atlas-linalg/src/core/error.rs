use atlas_ndarray::AtlasNdError;
use thiserror::Error;

pub type AtlasLinalgResult<T> = Result<T, AtlasLinalgError>;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AtlasLinalgError {
    #[error("invalid operand rank for {op}: left {left}, right {right}")]
    InvalidOperandRank { op: &'static str, left: usize, right: usize },

    #[error("shape mismatch for {op}: left {left:?}, right {right:?}: {reason}")]
    ShapeMismatch { op: &'static str, left: Vec<usize>, right: Vec<usize>, reason: &'static str },

    #[error("invalid input rank for {op}: expected {expected}, got rank {rank}")]
    InvalidInputRank { op: &'static str, expected: &'static str, rank: usize },

    #[error("invalid input shape for {op}: {shape:?}: {reason}")]
    InvalidInputShape { op: &'static str, shape: Vec<usize>, reason: &'static str },

    #[error("invalid {factor} factor for {op}: {reason}")]
    InvalidFactor { op: &'static str, factor: &'static str, reason: &'static str },

    #[error("non-finite input for {op}")]
    NonFiniteInput { op: &'static str },

    #[error("singular matrix for {op}: pivot {pivot}")]
    SingularMatrix { op: &'static str, pivot: usize },

    #[error("rank-deficient matrix for {op}: column {column}")]
    RankDeficientMatrix { op: &'static str, column: usize },

    #[error("matrix is not positive definite for {op}: diagonal {index}")]
    NotPositiveDefinite { op: &'static str, index: usize },

    #[error("invalid argument for {op}: {reason}")]
    InvalidArgument { op: &'static str, reason: &'static str },

    #[error("iteration limit reached for {op}: {iterations}")]
    IterationLimit { op: &'static str, iterations: usize },

    #[error(transparent)]
    NdArray(#[from] AtlasNdError),
}

#[cfg(test)]
mod tests {
    use super::AtlasLinalgError;

    #[test]
    fn error_messages_follow_consistent_style() {
        assert_eq!(
            AtlasLinalgError::ShapeMismatch {
                op: "matmul",
                left: vec![2, 3],
                right: vec![4, 2],
                reason: "left matrix column count must match right matrix row count",
            }
            .to_string(),
            "shape mismatch for matmul: left [2, 3], right [4, 2]: left matrix column count must match right matrix row count"
        );
        assert_eq!(
            AtlasLinalgError::SingularMatrix { op: "lu", pivot: 1 }.to_string(),
            "singular matrix for lu: pivot 1"
        );
    }
}
