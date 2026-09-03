mod core;
mod descriptive;

pub use core::error::{AtlasStatsError, AtlasStatsResult};
pub use core::operand::StatsOperand;
pub use descriptive::axis::{stddev_axis, variance_axis};
pub use descriptive::correlation::correlation;
pub use descriptive::covariance::covariance;
pub use descriptive::stddev::stddev;
pub use descriptive::variance::variance;
