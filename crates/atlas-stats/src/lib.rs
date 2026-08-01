mod core;
mod descriptive;

pub use core::error::{AtlasStatsError, AtlasStatsResult};
pub use core::operand::StatsOperand;
pub use descriptive::correlation::correlation;
pub use descriptive::covariance::covariance;
pub use descriptive::stddev::stddev;
pub use descriptive::variance::variance;
