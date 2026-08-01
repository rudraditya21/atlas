use atlas_ndarray::{NDArray, Numeric};
use num_traits::Float;
use rand::distributions::Distribution;
use rand_distr::StandardNormal;

use crate::{
    core::{AtlasRandomResult, element_count},
    rng::RandomSource,
};

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

#[cfg(test)]
mod tests {
    use crate::{AtlasRandomError, AtlasRng, normal};

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
    fn normal_reports_expected_validation_errors() {
        let mut rng = AtlasRng::seed_from_u64(23);

        assert!(matches!(
            normal([2], 0.0_f64, 0.0, &mut rng).unwrap_err(),
            AtlasRandomError::InvalidArgument { op: "normal", .. }
        ));
    }

    #[test]
    fn normal_supports_empty_shapes() {
        let mut rng = AtlasRng::seed_from_u64(29);
        let empty = normal([0], 0.0_f32, 1.0_f32, &mut rng).unwrap();

        assert_eq!(empty.shape(), &[0]);
        assert_eq!(empty.len(), 0);
    }
}
