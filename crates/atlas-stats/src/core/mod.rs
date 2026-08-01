pub(crate) mod error;
pub(crate) mod operand;
pub(crate) mod values;

pub use error::{AtlasStatsError, AtlasStatsResult};
pub use operand::StatsOperand;
pub(crate) use values::{collect_values, mean_of, validate_vector_pair};
