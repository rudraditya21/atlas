pub(crate) mod error;
pub(crate) mod sampling;

pub(crate) use error::{AtlasRandomError, AtlasRandomResult};
pub(crate) use sampling::{sample_ndarray, validate_normal_parameters, validate_uniform_bounds};
