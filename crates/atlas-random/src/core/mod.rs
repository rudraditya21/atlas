pub(crate) mod error;
pub(crate) mod sampling;

pub(crate) use error::{AtlasRandomError, AtlasRandomResult};
pub(crate) use sampling::{
    element_count, sample_ndarray, validate_bernoulli_probability, validate_normal_parameters,
    validate_uniform_bounds,
};
