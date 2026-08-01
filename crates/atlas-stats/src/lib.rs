mod descriptive;
mod error;
mod operand;

pub use descriptive::{correlation, covariance, stddev, variance};
pub use error::{AtlasStatsError, AtlasStatsResult};
pub use operand::StatsOperand;
