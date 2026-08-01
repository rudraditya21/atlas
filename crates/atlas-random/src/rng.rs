use core::cmp::Ordering;

use atlas_ndarray::Numeric;
use num_traits::Float;
use rand::{
    SeedableRng,
    distributions::{Distribution, Uniform, uniform::SampleUniform},
    rngs::StdRng,
};
use rand_distr::{Normal, StandardNormal};

use crate::error::{AtlasRandomError, AtlasRandomResult};

pub trait RandomSource {
    fn sample_uniform<T>(&mut self, low: T, high: T) -> AtlasRandomResult<T>
    where
        T: Numeric + SampleUniform + PartialOrd;

    fn sample_normal<T>(&mut self, mean: T, stddev: T) -> AtlasRandomResult<T>
    where
        T: Numeric + Float,
        StandardNormal: Distribution<T>;
}

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
        if low.partial_cmp(&high) != Some(Ordering::Less) {
            return Err(AtlasRandomError::InvalidArgument {
                op: "uniform",
                reason: "low must be strictly less than high",
            });
        }

        let distribution = Uniform::new(low, high);

        Ok(distribution.sample(&mut self.inner))
    }

    fn sample_normal<T>(&mut self, mean: T, stddev: T) -> AtlasRandomResult<T>
    where
        T: Numeric + Float,
        StandardNormal: Distribution<T>,
    {
        if !mean.is_finite() || !stddev.is_finite() {
            return Err(AtlasRandomError::InvalidArgument {
                op: "normal",
                reason: "mean and stddev must be finite",
            });
        }

        if stddev <= T::zero() {
            return Err(AtlasRandomError::InvalidArgument {
                op: "normal",
                reason: "stddev must be strictly positive",
            });
        }

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
