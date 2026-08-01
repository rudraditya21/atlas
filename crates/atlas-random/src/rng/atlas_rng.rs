use atlas_ndarray::Numeric;
use num_traits::Float;
use rand::{
    SeedableRng,
    distributions::{Distribution, Uniform, uniform::SampleUniform},
    rngs::StdRng,
};
use rand_distr::{Normal, StandardNormal};

use crate::core::{
    AtlasRandomError, AtlasRandomResult, validate_normal_parameters, validate_uniform_bounds,
};

use super::random_source::RandomSource;

#[derive(Clone, Debug)]
pub struct AtlasRng {
    inner: StdRng,
}

impl AtlasRng {
    pub fn new() -> Self {
        Self { inner: StdRng::from_entropy() }
    }

    pub fn seed_from_u64(seed: u64) -> Self {
        Self { inner: StdRng::seed_from_u64(seed) }
    }
}

impl Default for AtlasRng {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomSource for AtlasRng {
    fn sample_uniform<T>(&mut self, low: T, high: T) -> AtlasRandomResult<T>
    where
        T: Numeric + SampleUniform + PartialOrd,
    {
        validate_uniform_bounds(low, high)?;

        let distribution = Uniform::new(low, high);

        Ok(distribution.sample(&mut self.inner))
    }

    fn sample_normal<T>(&mut self, mean: T, stddev: T) -> AtlasRandomResult<T>
    where
        T: Numeric + Float,
        StandardNormal: Distribution<T>,
    {
        validate_normal_parameters(mean, stddev)?;

        let distribution = Normal::new(mean, stddev).map_err(|_| {
            AtlasRandomError::DistributionInitializationFailed {
                op: "normal",
                reason: "failed to build normal distribution",
            }
        })?;

        Ok(distribution.sample(&mut self.inner))
    }
}

#[cfg(test)]
mod tests {
    use super::{AtlasRng, RandomSource};

    #[test]
    fn seeded_rng_is_deterministic() {
        let mut left = AtlasRng::seed_from_u64(7);
        let mut right = AtlasRng::seed_from_u64(7);

        let left_uniform = left.sample_uniform(0.0_f64, 1.0).unwrap();
        let right_uniform = right.sample_uniform(0.0_f64, 1.0).unwrap();
        let left_normal = left.sample_normal(0.0_f64, 1.0).unwrap();
        let right_normal = right.sample_normal(0.0_f64, 1.0).unwrap();

        assert_eq!(left_uniform, right_uniform);
        assert_eq!(left_normal, right_normal);
    }
}
