pub(crate) mod error;
pub(crate) mod operand;
pub(crate) mod values;

pub(crate) use error::{AtlasStatsError, AtlasStatsResult};
pub(crate) use operand::StatsOperand;
pub(crate) use values::{
    mean, try_for_each_f64, try_for_each_vector_pair_f64, validate_non_empty,
    validate_non_empty_vector_pair,
};
