pub mod error;
pub mod rng;
pub mod sampling;

pub use error::{AtlasRandomError, AtlasRandomResult};
pub use rng::{AtlasRng, RandomSource};
pub use sampling::{normal, uniform};
