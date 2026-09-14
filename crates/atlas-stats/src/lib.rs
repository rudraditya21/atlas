mod core;
mod descriptive;

pub use core::{
    error::{AtlasStatsError, AtlasStatsResult},
    operand::StatsOperand,
};

pub use descriptive::{
    axis::{
        stddev_axis, stddev_axis_ddof, stddev_axis_keepdims, stddev_axis_keepdims_ddof,
        variance_axis, variance_axis_ddof, variance_axis_keepdims, variance_axis_keepdims_ddof,
    },
    correlation::correlation,
    correlation_matrix::correlation_matrix,
    covariance::{covariance, covariance_axis, covariance_axis_ddof, covariance_ddof},
    covariance_matrix::{covariance_matrix, covariance_matrix_ddof},
    kurtosis::kurtosis,
    quantile::{
        median, median_axis, median_axis_keepdims, quantile, quantile_axis, quantile_axis_keepdims,
    },
    skewness::skewness,
    stddev::{stddev, stddev_ddof},
    variance::{variance, variance_ddof},
    weighted::{weighted_covariance, weighted_mean, weighted_variance},
};
