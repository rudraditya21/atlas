use atlas_ndarray::Numeric;
use num_traits::Float;
use rand::distributions::{Distribution, uniform::SampleUniform};
use rand_distr::StandardNormal;

use crate::core::AtlasRandomResult;

pub trait RandomSource {
    fn sample_uniform<T>(&mut self, low: T, high: T) -> AtlasRandomResult<T>
    where
        T: Numeric + SampleUniform + PartialOrd;

    fn fill_uniform<T>(&mut self, low: T, high: T, output: &mut [T]) -> AtlasRandomResult<()>
    where
        T: Numeric + SampleUniform + PartialOrd;

    fn sample_normal<T>(&mut self, mean: T, stddev: T) -> AtlasRandomResult<T>
    where
        T: Numeric + Float,
        StandardNormal: Distribution<T>;

    fn fill_normal<T>(&mut self, mean: T, stddev: T, output: &mut [T]) -> AtlasRandomResult<()>
    where
        T: Numeric + Float,
        StandardNormal: Distribution<T>;
}
