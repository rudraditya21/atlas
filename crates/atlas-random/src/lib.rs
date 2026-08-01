mod core;
mod distributions;
mod rng;

pub use core::{AtlasRandomError, AtlasRandomResult};
pub use distributions::{normal, uniform};
pub use rng::{AtlasRng, RandomSource};
