use atlas_ndarray::{ArrayElement, NDArray, OperandMetadata};

use crate::{AtlasMlError, AtlasMlResult, core::validation::validate_supervised_training_inputs};

const OP: &str = "train_test_split";

/// Owned train and test partitions that preserve feature-target row alignment.
pub struct TrainTestSplit<T: ArrayElement> {
    train_features: NDArray<f64>,
    train_targets: NDArray<T>,
    test_features: NDArray<f64>,
    test_targets: NDArray<T>,
}

impl<T: ArrayElement> TrainTestSplit<T> {
    pub fn train_features(&self) -> &NDArray<f64> {
        &self.train_features
    }

    pub fn train_targets(&self) -> &NDArray<T> {
        &self.train_targets
    }

    pub fn test_features(&self) -> &NDArray<f64> {
        &self.test_features
    }

    pub fn test_targets(&self) -> &NDArray<T> {
        &self.test_targets
    }
}

/// Splits supervised data into deterministic seeded train and test partitions.
///
/// `test_ratio` must be finite and within `0.0..=1.0`. The test partition size
/// is the nearest integer to `sample_count * test_ratio`.
pub fn train_test_split<F, Targets, Y>(
    features: &F,
    targets: &Targets,
    test_ratio: f64,
    seed: u64,
) -> AtlasMlResult<TrainTestSplit<Y>>
where
    F: OperandMetadata<f64> + ?Sized,
    Targets: OperandMetadata<Y> + ?Sized,
    Y: ArrayElement,
{
    validate_supervised_training_inputs(features, targets, OP)?;
    if !test_ratio.is_finite() || !(0.0..=1.0).contains(&test_ratio) {
        return Err(AtlasMlError::InvalidArgument {
            op: OP,
            reason: "test ratio must be finite and between 0.0 and 1.0",
        });
    }

    let sample_count = features.shape()[0];
    let feature_count = features.shape()[1];
    let test_count = (sample_count as f64 * test_ratio).round() as usize;
    let mut indices = (0..sample_count).collect::<Vec<_>>();
    shuffle(&mut indices, seed);
    let (test_indices, train_indices) = indices.split_at(test_count);

    Ok(TrainTestSplit {
        train_features: select_feature_rows(features, train_indices, feature_count)?,
        train_targets: select_target_rows(targets, train_indices)?,
        test_features: select_feature_rows(features, test_indices, feature_count)?,
        test_targets: select_target_rows(targets, test_indices)?,
    })
}

fn select_feature_rows<F>(
    features: &F,
    indices: &[usize],
    feature_count: usize,
) -> AtlasMlResult<NDArray<f64>>
where
    F: OperandMetadata<f64> + ?Sized,
{
    let mut data = Vec::with_capacity(indices.len() * feature_count);
    for &sample_index in indices {
        let row_offset = features.offset() + sample_index * features.strides()[0];
        for feature_index in 0..feature_count {
            data.push(features.data()[row_offset + feature_index * features.strides()[1]]);
        }
    }

    Ok(NDArray::from_shape_vec([indices.len(), feature_count], data)?)
}

fn select_target_rows<T, Y>(targets: &T, indices: &[usize]) -> AtlasMlResult<NDArray<Y>>
where
    T: OperandMetadata<Y> + ?Sized,
    Y: ArrayElement,
{
    let data: Vec<Y> = indices
        .iter()
        .map(|&sample_index| targets.data()[targets.offset() + sample_index * targets.strides()[0]])
        .collect();

    Ok(NDArray::from_shape_vec([indices.len()], data)?)
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

    use super::train_test_split;
    use crate::AtlasMlError;

    #[test]
    fn splits_requested_sizes_without_breaking_row_alignment() {
        let features = NDArray::from_shape_vec(
            [5, 2],
            vec![0.0_f64, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0, 4.0, 14.0],
        )
        .unwrap();
        let targets = NDArray::from_shape_vec([5], vec![0_usize, 1, 2, 3, 4]).unwrap();
        let split = train_test_split(&features, &targets, 0.4, 7).unwrap();

        assert_eq!(split.train_features().shape(), &[3, 2]);
        assert_eq!(split.test_features().shape(), &[2, 2]);
        assert_aligned(split.train_features(), split.train_targets());
        assert_aligned(split.test_features(), split.test_targets());
    }

    #[test]
    fn seeded_splits_are_reproducible() {
        let features = NDArray::from_shape_vec([4, 1], vec![0.0_f64, 1.0, 2.0, 3.0]).unwrap();
        let targets = NDArray::from_shape_vec([4], vec![0_usize, 1, 2, 3]).unwrap();
        let first = train_test_split(&features, &targets, 0.5, 42).unwrap();
        let second = train_test_split(&features, &targets, 0.5, 42).unwrap();

        assert_eq!(first.train_features().data(), second.train_features().data());
        assert_eq!(first.train_targets().data(), second.train_targets().data());
        assert_eq!(first.test_features().data(), second.test_features().data());
        assert_eq!(first.test_targets().data(), second.test_targets().data());
    }

    #[test]
    fn supports_zero_sized_partitions() {
        let features = NDArray::from_shape_vec([2, 1], vec![0.0_f64, 1.0]).unwrap();
        let targets = NDArray::from_shape_vec([2], vec![0_usize, 1]).unwrap();
        let no_test = train_test_split(&features, &targets, 0.0, 1).unwrap();
        let no_train = train_test_split(&features, &targets, 1.0, 1).unwrap();

        assert_eq!(no_test.train_features().shape(), &[2, 1]);
        assert_eq!(no_test.test_features().shape(), &[0, 1]);
        assert_eq!(no_train.train_targets().shape(), &[0]);
        assert_eq!(no_train.test_targets().shape(), &[2]);
    }

    #[test]
    fn rejects_invalid_test_ratios() {
        let features = NDArray::from_shape_vec([1, 1], vec![0.0_f64]).unwrap();
        let targets = NDArray::from_shape_vec([1], vec![0_usize]).unwrap();

        for ratio in [-0.1, 1.1, f64::NAN] {
            assert_eq!(
                train_test_split(&features, &targets, ratio, 0).map(|_| ()),
                Err(AtlasMlError::InvalidArgument {
                    op: "train_test_split",
                    reason: "test ratio must be finite and between 0.0 and 1.0",
                })
            );
        }
    }

    #[test]
    fn preserves_logical_view_rows() {
        let features =
            NDArray::from_shape_vec([2, 4], vec![0.0_f64, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0])
                .unwrap();
        let targets = NDArray::from_shape_vec([4], vec![0_usize, 1, 2, 3]).unwrap();
        let split =
            train_test_split(&features.view().transpose(), &targets.view(), 0.5, 5).unwrap();

        assert_aligned(split.train_features(), split.train_targets());
        assert_aligned(split.test_features(), split.test_targets());
    }

    fn assert_aligned(features: &NDArray<f64>, targets: &NDArray<usize>) {
        for (row, &target) in features.data().chunks_exact(features.shape()[1]).zip(targets.data())
        {
            assert_eq!(row[0], target as f64);
        }
    }
}
