use num_traits::{Num, NumAssign};

pub trait Numeric: Num + NumAssign + Copy + Clone + Send + Sync + std::fmt::Debug {}

impl<T> Numeric for T where T: Num + NumAssign + Copy + Clone + Send + Sync + std::fmt::Debug {}
