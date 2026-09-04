use atlas_ndarray::NDArray;

use crate::{
    core::{AtlasRandomResult, element_count},
    rng::random_source::RandomSource,
};

/// Samples boolean values that are `true` with probability `probability`.
pub fn bernoulli<S, R>(shape: S, probability: f64, rng: &mut R) -> AtlasRandomResult<NDArray<bool>>
where
    S: AsRef<[usize]>,
    R: RandomSource,
{
    let shape = shape.as_ref().to_vec();
    let mut data = vec![false; element_count(&shape)?];
    rng.fill_bernoulli(probability, &mut data)?;
    NDArray::from_shape_vec(shape, data).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use crate::{AtlasRandomError, AtlasRng, bernoulli};

    #[test]
    fn bernoulli_is_seeded_and_handles_probability_boundaries() {
        let mut left = AtlasRng::seed_from_u64(17);
        let mut right = AtlasRng::seed_from_u64(17);

        assert_eq!(
            bernoulli([8], 0.4, &mut left).unwrap().data(),
            bernoulli([8], 0.4, &mut right).unwrap().data()
        );
        assert!(bernoulli([3], 0.0, &mut left).unwrap().data().iter().all(|value| !value));
        assert!(bernoulli([3], 1.0, &mut left).unwrap().data().iter().all(|value| *value));
    }

    #[test]
    fn bernoulli_rejects_invalid_probabilities() {
        let mut rng = AtlasRng::seed_from_u64(19);
        assert!(matches!(
            bernoulli([1], f64::NAN, &mut rng).unwrap_err(),
            AtlasRandomError::InvalidArgument { op: "bernoulli", .. }
        ));
        assert!(matches!(
            bernoulli([1], 1.1, &mut rng).unwrap_err(),
            AtlasRandomError::InvalidArgument { op: "bernoulli", .. }
        ));
    }
}
