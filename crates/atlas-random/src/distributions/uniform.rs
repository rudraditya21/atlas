use atlas_ndarray::{NDArray, Numeric};
use rand::distributions::uniform::SampleUniform;

use crate::{
    core::{AtlasRandomResult, sample_ndarray},
    rng::random_source::RandomSource,
};

pub fn uniform<T, S, R>(shape: S, low: T, high: T, rng: &mut R) -> AtlasRandomResult<NDArray<T>>
where
    T: Numeric + SampleUniform + PartialOrd,
    S: AsRef<[usize]>,
    R: RandomSource,
{
    sample_ndarray(shape, |data| rng.fill_uniform(low, high, data))
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::AtlasNdError;

    use crate::{AtlasRandomError, AtlasRng, uniform};

    #[test]
    fn uniform_sampling_generates_bounded_ndarrays() {
        let mut rng = AtlasRng::seed_from_u64(11);
        let sampled = uniform([2, 3], -1.0_f64, 1.0, &mut rng).unwrap();

        assert_eq!(sampled.shape(), &[2, 3]);
        assert_eq!(sampled.len(), 6);
        assert!(sampled.data().iter().all(|value| *value >= -1.0 && *value < 1.0));
    }

    #[test]
    fn uniform_reports_expected_validation_errors() {
        let mut rng = AtlasRng::seed_from_u64(23);

        assert!(matches!(
            uniform([2], 1.0_f64, 1.0, &mut rng).unwrap_err(),
            AtlasRandomError::InvalidArgument { op: "uniform", .. }
        ));
    }

    #[test]
    fn uniform_supports_scalar_shapes() {
        let mut rng = AtlasRng::seed_from_u64(29);
        let scalar = uniform([], 0_i32, 10_i32, &mut rng).unwrap();

        assert_eq!(scalar.shape(), &[] as &[usize]);
        assert_eq!(scalar.len(), 1);
    }

    #[test]
    fn uniform_reports_shape_overflow_explicitly() {
        let mut rng = AtlasRng::seed_from_u64(31);

        assert_eq!(
            uniform([usize::MAX, 2], 0_i32, 10_i32, &mut rng).unwrap_err(),
            AtlasRandomError::NdArray(AtlasNdError::ShapeOverflow {
                op: "element count",
                shape: vec![usize::MAX, 2],
            })
        );
    }
}
