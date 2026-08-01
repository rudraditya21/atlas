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
