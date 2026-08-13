use atlas_ndarray::NDArray;

use crate::{
    core::AtlasRandomResult, distributions::normal::normal, rng::random_source::RandomSource,
};

pub fn randn<S, R>(shape: S, rng: &mut R) -> AtlasRandomResult<NDArray<f64>>
where
    S: AsRef<[usize]>,
    R: RandomSource,
{
    normal(shape, 0.0_f64, 1.0_f64, rng)
}

#[cfg(test)]
mod tests {
    use crate::{AtlasRng, randn};

    #[test]
    fn randn_generates_finite_standard_normal_f64_arrays() {
        let mut rng = AtlasRng::seed_from_u64(47);
        let sampled = randn([2, 2], &mut rng).unwrap();

        assert_eq!(sampled.shape(), &[2, 2]);
        assert!(sampled.data().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn randn_is_seeded_and_supports_empty_shapes() {
        let mut left = AtlasRng::seed_from_u64(53);
        let mut right = AtlasRng::seed_from_u64(53);

        let lhs = randn([0], &mut left).unwrap();
        let rhs = randn([0], &mut right).unwrap();

        assert_eq!(lhs.shape(), &[0]);
        assert_eq!(lhs.data(), rhs.data());
    }
}
