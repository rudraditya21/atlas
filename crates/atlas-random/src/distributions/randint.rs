use atlas_ndarray::{NDArray, Numeric};
use rand::distributions::uniform::SampleUniform;

use crate::{
    core::AtlasRandomResult, distributions::uniform::uniform, rng::random_source::RandomSource,
};

mod sealed {
    pub trait Sealed {}
    macro_rules! impl_sealed { ($($ty:ty),+ $(,)?) => { $(impl Sealed for $ty {})+ }; }
    impl_sealed!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);
}

/// Integer dtypes supported by [`randint`].
pub trait IntegerRange: Numeric + SampleUniform + PartialOrd + sealed::Sealed {}

macro_rules! impl_integer_range { ($($ty:ty),+ $(,)?) => { $(impl IntegerRange for $ty {})+ }; }
impl_integer_range!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

/// Samples integers uniformly from the half-open range `[low, high)`.
pub fn randint<T, S, R>(shape: S, low: T, high: T, rng: &mut R) -> AtlasRandomResult<NDArray<T>>
where
    T: IntegerRange,
    S: AsRef<[usize]>,
    R: RandomSource,
{
    uniform(shape, low, high, rng)
}

#[cfg(test)]
mod tests {
    use crate::{AtlasRandomError, AtlasRng, randint};

    #[test]
    fn randint_samples_the_requested_half_open_range() {
        let mut rng = AtlasRng::seed_from_u64(31);
        let sampled = randint([128], -3_i32, 5_i32, &mut rng).unwrap();

        assert!(sampled.data().iter().all(|value| (-3..5).contains(value)));
    }

    #[test]
    fn randint_is_seeded_and_validates_ranges() {
        let mut left = AtlasRng::seed_from_u64(37);
        let mut right = AtlasRng::seed_from_u64(37);

        assert_eq!(
            randint([8], 0_u64, 10_u64, &mut left).unwrap().data(),
            randint([8], 0_u64, 10_u64, &mut right).unwrap().data()
        );
        assert!(matches!(
            randint([1], 4_i32, 4_i32, &mut left).unwrap_err(),
            AtlasRandomError::InvalidArgument { op: "uniform", .. }
        ));
    }
}
