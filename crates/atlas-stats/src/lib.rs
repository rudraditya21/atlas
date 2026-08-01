mod core;
mod descriptive;

pub use core::{AtlasStatsError, AtlasStatsResult, StatsOperand};
pub use descriptive::{correlation, covariance, stddev, variance};
