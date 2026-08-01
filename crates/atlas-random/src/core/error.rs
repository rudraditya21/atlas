use atlas_ndarray::AtlasNdError;
use thiserror::Error;

pub type AtlasRandomResult<T> = Result<T, AtlasRandomError>;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AtlasRandomError {
    #[error("invalid argument for {op}: {reason}")]
    InvalidArgument { op: &'static str, reason: &'static str },

    #[error("distribution initialization failed for {op}: {reason}")]
    DistributionInitializationFailed { op: &'static str, reason: &'static str },

    #[error(transparent)]
    NdArray(#[from] AtlasNdError),
}

#[cfg(test)]
mod tests {
    use super::AtlasRandomError;

    #[test]
    fn error_messages_follow_consistent_style() {
        assert_eq!(
            AtlasRandomError::InvalidArgument {
                op: "normal",
                reason: "stddev must be strictly positive",
            }
            .to_string(),
            "invalid argument for normal: stddev must be strictly positive"
        );
        assert_eq!(
            AtlasRandomError::DistributionInitializationFailed {
                op: "normal",
                reason: "failed to build normal distribution",
            }
            .to_string(),
            "distribution initialization failed for normal: failed to build normal distribution"
        );
    }
}
