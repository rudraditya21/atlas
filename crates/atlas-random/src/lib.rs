mod core;
mod distributions;
mod rng;

pub use core::error::{AtlasRandomError, AtlasRandomResult};
pub use distributions::bernoulli::bernoulli;
pub use distributions::normal::normal;
pub use distributions::rand::rand;
pub use distributions::randint::IntegerRange;
pub use distributions::randint::randint;
pub use distributions::randn::randn;
pub use distributions::uniform::uniform;
pub use rng::atlas_rng::AtlasRng;
pub use rng::random_source::RandomSource;
