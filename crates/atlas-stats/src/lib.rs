pub mod descriptive;
pub mod error;
pub mod operand;

pub use descriptive::{correlation, covariance, stddev, variance};
pub use error::{AtlasStatsError, AtlasStatsResult};
pub use operand::StatsOperand;
