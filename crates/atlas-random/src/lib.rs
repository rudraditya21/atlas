mod error;
mod rng;
mod sampling;

pub use error::{AtlasRandomError, AtlasRandomResult};
pub use rng::{AtlasRng, RandomSource};
pub use sampling::{normal, uniform};
