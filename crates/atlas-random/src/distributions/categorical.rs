use atlas_ndarray::NDArray;

use crate::{
    core::{AtlasRandomError, AtlasRandomResult, sample_ndarray},
    rng::random_source::RandomSource,
};

/// Samples category indices according to non-negative probability weights.
pub fn categorical<S, R>(
    shape: S,
    weights: &[f64],
    rng: &mut R,
) -> AtlasRandomResult<NDArray<usize>>
where
    S: AsRef<[usize]>,
    R: RandomSource,
{
    let total = validate_weights(weights)?;
    sample_ndarray(shape, |output| {
        for value in output {
            let sample = rng.sample_uniform(0.0_f64, total)?;
            let mut cumulative = 0.0;
            let mut category = weights.len() - 1;
            for (index, weight) in weights.iter().enumerate() {
                cumulative += weight;
                if sample < cumulative {
                    category = index;
                    break;
                }
            }
            *value = category;
        }
        Ok(())
    })
}

fn validate_weights(weights: &[f64]) -> AtlasRandomResult<f64> {
    if weights.is_empty() {
        return Err(AtlasRandomError::InvalidArgument {
            op: "categorical",
            reason: "weights must not be empty",
        });
    }

    let mut total = 0.0;
    for &weight in weights {
        if !weight.is_finite() || weight < 0.0 {
            return Err(AtlasRandomError::InvalidArgument {
                op: "categorical",
                reason: "weights must be finite and non-negative",
            });
        }
        total += weight;
    }

    if total.is_finite() && total > 0.0 {
        Ok(total)
    } else {
        Err(AtlasRandomError::InvalidArgument {
            op: "categorical",
            reason: "weights must have a positive finite sum",
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{AtlasRandomError, AtlasRng, categorical};

    #[test]
    fn categorical_is_seeded_and_returns_valid_indices() {
        let mut left = AtlasRng::seed_from_u64(59);
        let mut right = AtlasRng::seed_from_u64(59);
        let weights = [1.0, 2.0, 3.0];
        let left = categorical([32], &weights, &mut left).unwrap();
        let right = categorical([32], &weights, &mut right).unwrap();

        assert_eq!(left.data(), right.data());
        assert!(left.data().iter().all(|index| *index < weights.len()));
    }

    #[test]
    fn categorical_rejects_invalid_weights() {
        let mut rng = AtlasRng::seed_from_u64(61);
        assert!(matches!(
            categorical([1], &[], &mut rng).unwrap_err(),
            AtlasRandomError::InvalidArgument { op: "categorical", .. }
        ));
        assert!(matches!(
            categorical([1], &[1.0, -1.0], &mut rng).unwrap_err(),
            AtlasRandomError::InvalidArgument { op: "categorical", .. }
        ));
        assert!(matches!(
            categorical([1], &[0.0, 0.0], &mut rng).unwrap_err(),
            AtlasRandomError::InvalidArgument { op: "categorical", .. }
        ));
    }
}
