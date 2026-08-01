use atlas_ndarray::Numeric;
use num_traits::Float;
use rand::{
    RngCore, SeedableRng,
    distributions::{Distribution, Uniform, uniform::SampleUniform},
    rngs::StdRng,
};
use rand_distr::{Normal, StandardNormal};
use rayon::prelude::*;

use crate::core::{
    AtlasRandomError, AtlasRandomResult, validate_normal_parameters, validate_uniform_bounds,
};

use super::random_source::RandomSource;

const PARALLEL_SAMPLING_THRESHOLD: usize = 1 << 18;
const PARALLEL_SAMPLING_CHUNK_LEN: usize = 1 << 14;

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

    fn should_parallelize_fill(len: usize) -> bool {
        len >= PARALLEL_SAMPLING_THRESHOLD && rayon::current_num_threads() > 1
    }

    fn chunk_seeds(&mut self, len: usize) -> Vec<u64> {
        let chunk_count = len.div_ceil(PARALLEL_SAMPLING_CHUNK_LEN);
        let mut seeds = Vec::with_capacity(chunk_count);

        for _ in 0..chunk_count {
            seeds.push(self.inner.next_u64());
        }

        seeds
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

    fn fill_uniform<T>(&mut self, low: T, high: T, output: &mut [T]) -> AtlasRandomResult<()>
    where
        T: Numeric + SampleUniform + PartialOrd,
    {
        validate_uniform_bounds(low, high)?;

        if output.is_empty() {
            return Ok(());
        }

        if Self::should_parallelize_fill(output.len()) {
            let chunk_seeds = self.chunk_seeds(output.len());

            output
                .par_chunks_mut(PARALLEL_SAMPLING_CHUNK_LEN)
                .zip(chunk_seeds.into_par_iter())
                .for_each(|(chunk, seed)| {
                    let distribution = Uniform::new(low, high);
                    let mut rng = StdRng::seed_from_u64(seed);

                    for value in chunk.iter_mut() {
                        *value = distribution.sample(&mut rng);
                    }
                });

            return Ok(());
        }

        let distribution = Uniform::new(low, high);

        for value in output.iter_mut() {
            *value = distribution.sample(&mut self.inner);
        }

        Ok(())
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

    fn fill_normal<T>(&mut self, mean: T, stddev: T, output: &mut [T]) -> AtlasRandomResult<()>
    where
        T: Numeric + Float,
        StandardNormal: Distribution<T>,
    {
        validate_normal_parameters(mean, stddev)?;

        if output.is_empty() {
            return Ok(());
        }

        if Self::should_parallelize_fill(output.len()) {
            let chunk_seeds = self.chunk_seeds(output.len());

            output
                .par_chunks_mut(PARALLEL_SAMPLING_CHUNK_LEN)
                .zip(chunk_seeds.into_par_iter())
                .for_each(|(chunk, seed)| {
                    let distribution = Normal::new(mean, stddev)
                        .expect("validated normal parameters must initialize the distribution");
                    let mut rng = StdRng::seed_from_u64(seed);

                    for value in chunk.iter_mut() {
                        *value = distribution.sample(&mut rng);
                    }
                });

            return Ok(());
        }

        let distribution = Normal::new(mean, stddev).map_err(|_| {
            AtlasRandomError::DistributionInitializationFailed {
                op: "normal",
                reason: "failed to build normal distribution",
            }
        })?;

        for value in output.iter_mut() {
            *value = distribution.sample(&mut self.inner);
        }

        Ok(())
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

    #[test]
    fn large_parallel_fills_remain_repeatable_for_seeded_rngs() {
        let mut left = AtlasRng::seed_from_u64(11);
        let mut right = AtlasRng::seed_from_u64(11);
        let mut left_uniform = vec![0.0_f64; super::PARALLEL_SAMPLING_THRESHOLD];
        let mut right_uniform = vec![0.0_f64; super::PARALLEL_SAMPLING_THRESHOLD];
        let mut left_normal = vec![0.0_f64; super::PARALLEL_SAMPLING_THRESHOLD];
        let mut right_normal = vec![0.0_f64; super::PARALLEL_SAMPLING_THRESHOLD];

        left.fill_uniform(0.0_f64, 1.0, &mut left_uniform).unwrap();
        right.fill_uniform(0.0_f64, 1.0, &mut right_uniform).unwrap();
        left.fill_normal(0.0_f64, 1.0, &mut left_normal).unwrap();
        right.fill_normal(0.0_f64, 1.0, &mut right_normal).unwrap();

        assert_eq!(left_uniform, right_uniform);
        assert_eq!(left_normal, right_normal);
        assert_eq!(
            left.sample_uniform(0_i32, 10_i32).unwrap(),
            right.sample_uniform(0_i32, 10_i32).unwrap()
        );
        assert_eq!(
            left.sample_normal(0.0_f64, 1.0).unwrap(),
            right.sample_normal(0.0_f64, 1.0).unwrap()
        );
    }
}
