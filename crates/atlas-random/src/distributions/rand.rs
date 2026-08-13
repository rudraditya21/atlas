use atlas_ndarray::NDArray;

use crate::{
    core::AtlasRandomResult, distributions::uniform::uniform, rng::random_source::RandomSource,
};

pub fn rand<S, R>(shape: S, rng: &mut R) -> AtlasRandomResult<NDArray<f64>>
where
    S: AsRef<[usize]>,
    R: RandomSource,
{
    uniform(shape, 0.0_f64, 1.0_f64, rng)
}

#[cfg(test)]
mod tests {
    use crate::{AtlasRng, rand};

    #[test]
    fn rand_generates_unit_interval_f64_arrays() {
        let mut rng = AtlasRng::seed_from_u64(41);
        let sampled = rand([2, 3], &mut rng).unwrap();

        assert_eq!(sampled.shape(), &[2, 3]);
        assert!(sampled.data().iter().all(|value| *value >= 0.0 && *value < 1.0));
    }

    #[test]
    fn rand_is_seeded_and_supports_scalar_shapes() {
        let mut left = AtlasRng::seed_from_u64(43);
        let mut right = AtlasRng::seed_from_u64(43);

        let lhs = rand([], &mut left).unwrap();
        let rhs = rand([], &mut right).unwrap();

        assert_eq!(lhs.shape(), &[] as &[usize]);
        assert_eq!(lhs.data(), rhs.data());
    }
}
