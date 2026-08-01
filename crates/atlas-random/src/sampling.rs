use atlas_ndarray::{NDArray, Numeric};
use num_traits::Float;
use rand::distributions::{Distribution, uniform::SampleUniform};
use rand_distr::StandardNormal;

use crate::{error::AtlasRandomResult, rng::RandomSource};

pub fn uniform<T, S, R>(shape: S, low: T, high: T, rng: &mut R) -> AtlasRandomResult<NDArray<T>>
where
    T: Numeric + SampleUniform + PartialOrd,
    S: AsRef<[usize]>,
    R: RandomSource,
{
    let shape = shape.as_ref().to_vec();
    let len = element_count(&shape);
    let mut data = Vec::with_capacity(len);

    for _ in 0..len {
        data.push(rng.sample_uniform(low, high)?);
    }

    Ok(NDArray::from_shape_vec(shape, data)?)
}

pub fn normal<T, S, R>(shape: S, mean: T, stddev: T, rng: &mut R) -> AtlasRandomResult<NDArray<T>>
where
    T: Numeric + Float,
    S: AsRef<[usize]>,
    R: RandomSource,
    StandardNormal: Distribution<T>,
{
    let shape = shape.as_ref().to_vec();
    let len = element_count(&shape);
    let mut data = Vec::with_capacity(len);

    for _ in 0..len {
        data.push(rng.sample_normal(mean, stddev)?);
    }

    Ok(NDArray::from_shape_vec(shape, data)?)
}

fn element_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

#[cfg(test)]
mod tests {
    use crate::{AtlasRandomError, AtlasRng, normal, uniform};

    #[test]
    fn uniform_sampling_generates_bounded_ndarrays() {
        let mut rng = AtlasRng::seed_from_u64(11);
        let sampled = uniform([2, 3], -1.0_f64, 1.0, &mut rng).unwrap();

        assert_eq!(sampled.shape(), &[2, 3]);
        assert_eq!(sampled.len(), 6);
        assert!(sampled.data().iter().all(|value| *value >= -1.0 && *value < 1.0));
    }

    #[test]
    fn normal_sampling_is_deterministic_for_seeded_rngs() {
        let mut left = AtlasRng::seed_from_u64(19);
        let mut right = AtlasRng::seed_from_u64(19);

        let lhs = normal([4], 0.0_f64, 1.5, &mut left).unwrap();
        let rhs = normal([4], 0.0_f64, 1.5, &mut right).unwrap();

        assert_eq!(lhs.shape(), &[4]);
        assert_eq!(lhs.data(), rhs.data());
        assert!(lhs.data().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn sampling_reports_expected_validation_errors() {
        let mut rng = AtlasRng::seed_from_u64(23);

        assert!(matches!(
            uniform([2], 1.0_f64, 1.0, &mut rng).unwrap_err(),
            AtlasRandomError::InvalidArgument { op: "uniform", .. }
        ));
        assert!(matches!(
            normal([2], 0.0_f64, 0.0, &mut rng).unwrap_err(),
            AtlasRandomError::InvalidArgument { op: "normal", .. }
        ));
    }

    #[test]
    fn scalar_and_empty_shapes_are_supported() {
        let mut rng = AtlasRng::seed_from_u64(29);

        let scalar = uniform([], 0_i32, 10_i32, &mut rng).unwrap();
        let empty = normal([0], 0.0_f32, 1.0_f32, &mut rng).unwrap();

        assert_eq!(scalar.shape(), &[] as &[usize]);
        assert_eq!(scalar.len(), 1);
        assert_eq!(empty.shape(), &[0]);
        assert_eq!(empty.len(), 0);
    }
}
