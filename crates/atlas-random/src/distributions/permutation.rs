use atlas_ndarray::NDArray;

use crate::{core::AtlasRandomResult, rng::random_source::RandomSource};

/// Returns a uniformly shuffled, seeded permutation of `0..n`.
pub fn permutation<R: RandomSource>(n: usize, rng: &mut R) -> AtlasRandomResult<NDArray<usize>> {
    let mut values = (0..n).collect::<Vec<_>>();
    rng.shuffle(&mut values);
    NDArray::from_shape_vec([n], values).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use crate::{AtlasRng, permutation};

    #[test]
    fn permutation_is_seeded_and_contains_each_index_once() {
        let mut left = AtlasRng::seed_from_u64(41);
        let mut right = AtlasRng::seed_from_u64(41);
        let left = permutation(16, &mut left).unwrap();
        let right = permutation(16, &mut right).unwrap();
        let mut sorted = left.data().to_vec();
        sorted.sort_unstable();

        assert_eq!(left.data(), right.data());
        assert_eq!(sorted, (0..16).collect::<Vec<_>>());
    }

    #[test]
    fn permutation_supports_empty_inputs() {
        let mut rng = AtlasRng::seed_from_u64(43);
        assert!(permutation(0, &mut rng).unwrap().data().is_empty());
    }
}
