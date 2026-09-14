mod core;
mod distributions;
mod rng;

pub use core::error::{AtlasRandomError, AtlasRandomResult};

pub use distributions::{
    bernoulli::bernoulli,
    categorical::categorical,
    choice::{choice, choice_indices, choice_indices_with_replacement},
    normal::normal,
    permutation::permutation,
    rand::rand,
    randint::{IntegerRange, randint},
    randn::randn,
    shuffle::shuffle_axis,
    uniform::uniform,
};
pub use rng::{atlas_rng::AtlasRng, random_source::RandomSource};
