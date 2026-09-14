use atlas_ndarray::OperandMetadata;

use crate::{AtlasMlError, AtlasMlResult};

const OP: &str = "k_fold_split";

/// Training and validation sample indices for one deterministic K-fold partition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KFold {
    train_indices: Box<[usize]>,
    validation_indices: Box<[usize]>,
}

impl KFold {
    /// Returns the training sample indices for this fold.
    pub fn train_indices(&self) -> &[usize] {
        &self.train_indices
    }

    /// Returns the validation sample indices for this fold.
    pub fn validation_indices(&self) -> &[usize] {
        &self.validation_indices
    }
}

/// Produces deterministic seeded K-fold train and validation indices for rank-2 features.
pub fn k_fold_split<F>(features: &F, fold_count: usize, seed: u64) -> AtlasMlResult<Vec<KFold>>
where
    F: OperandMetadata<f64> + ?Sized,
{
    validate_inputs(features, fold_count)?;

    let sample_count = features.shape()[0];
    let mut indices = (0..sample_count).collect::<Vec<_>>();
    shuffle(&mut indices, seed);
    let base_fold_size = sample_count / fold_count;
    let extra_samples = sample_count % fold_count;
    let mut start = 0;
    let mut folds = Vec::with_capacity(fold_count);
    for fold_index in 0..fold_count {
        let fold_size = base_fold_size + if fold_index < extra_samples { 1 } else { 0 };
        let end = start + fold_size;
        let validation_indices = indices[start..end].to_vec().into();
        let mut train_indices = Vec::with_capacity(sample_count - fold_size);
        train_indices.extend_from_slice(&indices[..start]);
        train_indices.extend_from_slice(&indices[end..]);
        folds.push(KFold { train_indices: train_indices.into(), validation_indices });
        start = end;
    }

    Ok(folds)
}

fn validate_inputs<F>(features: &F, fold_count: usize) -> AtlasMlResult<()>
where
    F: OperandMetadata<f64> + ?Sized,
{
    if features.ndim() != 2 {
        return Err(AtlasMlError::InvalidInputRank {
            op: OP,
            expected: "a rank-2 [samples, features] matrix",
            rank: features.ndim(),
        });
    }

    let sample_count = features.shape()[0];
    if sample_count == 0 {
        return Err(AtlasMlError::EmptyInput { op: OP });
    }
    if fold_count < 2 {
        return Err(AtlasMlError::InvalidArgument {
            op: OP,
            reason: "fold count must be at least two",
        });
    }
    if fold_count > sample_count {
        return Err(AtlasMlError::InvalidArgument {
            op: OP,
            reason: "fold count must not exceed the number of samples",
        });
    }

    Ok(())
}

fn shuffle(indices: &mut [usize], seed: u64) {
    let mut state = seed;
    for upper_bound in (2..=indices.len()).rev() {
        let index = bounded_random(&mut state, upper_bound);
        indices.swap(upper_bound - 1, index);
    }
}

fn bounded_random(state: &mut u64, upper_bound: usize) -> usize {
    let upper_bound = upper_bound as u64;
    let threshold = upper_bound.wrapping_neg() % upper_bound;
    loop {
        *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        if *state >= threshold {
            return (*state % upper_bound) as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::k_fold_split;
    use crate::AtlasMlError;

    #[test]
    fn validation_folds_cover_each_sample_once() {
        let features = NDArray::from_shape_vec([5, 1], vec![0.0_f64; 5]).unwrap();
        let folds = k_fold_split(&features, 5, 7).unwrap();
        let mut validation_indices = folds
            .iter()
            .flat_map(|fold| fold.validation_indices().iter().copied())
            .collect::<Vec<_>>();
        validation_indices.sort_unstable();

        assert_eq!(validation_indices, vec![0, 1, 2, 3, 4]);
        for fold in &folds {
            assert_eq!(fold.train_indices().len(), 4);
            assert_eq!(fold.validation_indices().len(), 1);
            assert!(!fold.train_indices().contains(&fold.validation_indices()[0]));
        }
    }

    #[test]
    fn seeded_folds_are_reproducible() {
        let features = NDArray::from_shape_vec([4, 1], vec![0.0_f64; 4]).unwrap();

        assert_eq!(k_fold_split(&features, 2, 42), k_fold_split(&features, 2, 42));
    }

    #[test]
    fn distributes_uneven_samples_to_earlier_folds() {
        let features = NDArray::from_shape_vec([5, 1], vec![0.0_f64; 5]).unwrap();
        let folds = k_fold_split(&features, 2, 1).unwrap();

        assert_eq!(folds[0].validation_indices().len(), 3);
        assert_eq!(folds[1].validation_indices().len(), 2);
    }

    #[test]
    fn rejects_invalid_fold_counts() {
        let features = NDArray::from_shape_vec([3, 1], vec![0.0_f64; 3]).unwrap();

        assert_eq!(
            k_fold_split(&features, 1, 0),
            Err(AtlasMlError::InvalidArgument {
                op: "k_fold_split",
                reason: "fold count must be at least two"
            })
        );
        assert_eq!(
            k_fold_split(&features, 4, 0),
            Err(AtlasMlError::InvalidArgument {
                op: "k_fold_split",
                reason: "fold count must not exceed the number of samples",
            })
        );
    }

    #[test]
    fn supports_logical_feature_views() {
        let features = NDArray::from_shape_vec([2, 4], vec![0.0_f64; 8]).unwrap();
        let folds = k_fold_split(&features.view().transpose(), 2, 3).unwrap();
        let mut validation_indices = folds
            .iter()
            .flat_map(|fold| fold.validation_indices().iter().copied())
            .collect::<Vec<_>>();
        validation_indices.sort_unstable();

        assert_eq!(validation_indices, vec![0, 1, 2, 3]);
    }
}
