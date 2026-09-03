mod core;
mod descriptive;

pub use core::error::{AtlasStatsError, AtlasStatsResult};
pub use core::operand::StatsOperand;
pub use descriptive::axis::{stddev_axis, variance_axis};
pub use descriptive::correlation::correlation;
pub use descriptive::correlation_matrix::correlation_matrix;
pub use descriptive::covariance::covariance;
pub use descriptive::covariance_matrix::covariance_matrix;
pub use descriptive::quantile::{
    median, median_axis, median_axis_keepdims, quantile, quantile_axis, quantile_axis_keepdims,
};
pub use descriptive::stddev::stddev;
pub use descriptive::variance::variance;
pub use descriptive::weighted::{weighted_covariance, weighted_mean, weighted_variance};
