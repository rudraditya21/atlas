pub(crate) mod error;
pub(crate) mod sampling;

pub use error::{AtlasRandomError, AtlasRandomResult};
pub(crate) use sampling::{element_count, validate_normal_parameters, validate_uniform_bounds};
