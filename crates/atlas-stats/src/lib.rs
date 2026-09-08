mod core;
mod descriptive;

pub use core::{
    error::{AtlasStatsError, AtlasStatsResult},
    operand::StatsOperand,
};

pub use descriptive::{
    axis::{stddev_axis, variance_axis},
    correlation::correlation,
    correlation_matrix::correlation_matrix,
    covariance::covariance,
    covariance_matrix::covariance_matrix,
    quantile::{
        median, median_axis, median_axis_keepdims, quantile, quantile_axis, quantile_axis_keepdims,
    },
    stddev::stddev,
    variance::variance,
    weighted::{weighted_covariance, weighted_mean, weighted_variance},
};
