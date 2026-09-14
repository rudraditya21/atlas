use atlas_ndarray::OperandMetadata;

use crate::{
    AtlasMlError, AtlasMlResult,
    core::validation::{validate_binary_labels, validate_finite_feature_values},
};

const BINARY_GINI_SPLIT_OP: &str = "binary_gini_split";

/// Weighted binary Gini impurity and branch sizes for one threshold split.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinaryGiniSplit {
    impurity: f64,
    left_count: usize,
    right_count: usize,
}

impl BinaryGiniSplit {
    /// Returns the sample-count-weighted impurity across both branches.
    pub const fn impurity(&self) -> f64 {
        self.impurity
    }

    /// Returns the number of samples with feature value less than or equal to the threshold.
    pub const fn left_count(&self) -> usize {
        self.left_count
    }

    /// Returns the number of samples with feature value greater than the threshold.
    pub const fn right_count(&self) -> usize {
        self.right_count
    }
}

/// Evaluates the weighted binary Gini impurity for `feature <= threshold` and `feature > threshold`.
pub fn evaluate_binary_gini_split<F, L>(
    feature_values: &F,
    labels: &L,
    threshold: f64,
) -> AtlasMlResult<BinaryGiniSplit>
where
    F: OperandMetadata<f64> + ?Sized,
    L: OperandMetadata<usize> + ?Sized,
{
    validate_feature_values(feature_values)?;
    validate_binary_labels(labels, BINARY_GINI_SPLIT_OP)?;
    if feature_values.shape()[0] != labels.shape()[0] {
        return Err(AtlasMlError::ShapeMismatch {
            op: BINARY_GINI_SPLIT_OP,
            left: feature_values.shape().to_vec(),
            right: labels.shape().to_vec(),
            reason: "sample counts must match",
        });
    }
    if !threshold.is_finite() {
        return Err(AtlasMlError::NonFiniteInput { op: BINARY_GINI_SPLIT_OP });
    }

    let mut left_counts = [0_usize; 2];
    let mut right_counts = [0_usize; 2];
    for sample_index in 0..feature_values.shape()[0] {
        let label = label(labels, sample_index);
        let counts = if feature(feature_values, sample_index) <= threshold {
            &mut left_counts
        } else {
            &mut right_counts
        };
        counts[label] += 1;
    }

    let left_count = left_counts.iter().sum();
    let right_count = right_counts.iter().sum();
    if left_count == 0 || right_count == 0 {
        return Err(AtlasMlError::InvalidArgument {
            op: BINARY_GINI_SPLIT_OP,
            reason: "split must place samples on both sides of the threshold",
        });
    }

    let sample_count = feature_values.shape()[0] as f64;
    Ok(BinaryGiniSplit {
        impurity: left_count as f64 / sample_count * gini(&left_counts, left_count)
            + right_count as f64 / sample_count * gini(&right_counts, right_count),
        left_count,
        right_count,
    })
}

fn validate_feature_values<F>(feature_values: &F) -> AtlasMlResult<()>
where
    F: OperandMetadata<f64> + ?Sized,
{
    if feature_values.ndim() != 1 {
        return Err(AtlasMlError::InvalidInputRank {
            op: BINARY_GINI_SPLIT_OP,
            expected: "a rank-1 feature vector",
            rank: feature_values.ndim(),
        });
    }
    if feature_values.shape()[0] == 0 {
        return Err(AtlasMlError::EmptyInput { op: BINARY_GINI_SPLIT_OP });
    }

    validate_finite_feature_values(feature_values, BINARY_GINI_SPLIT_OP)
}

fn feature<F>(feature_values: &F, sample_index: usize) -> f64
where
    F: OperandMetadata<f64> + ?Sized,
{
    feature_values.data()[feature_values.offset() + sample_index * feature_values.strides()[0]]
}

fn label<L>(labels: &L, sample_index: usize) -> usize
where
    L: OperandMetadata<usize> + ?Sized,
{
    labels.data()[labels.offset() + sample_index * labels.strides()[0]]
}

fn gini(counts: &[usize; 2], count: usize) -> f64 {
    1.0 - counts.iter().map(|&class_count| (class_count as f64 / count as f64).powi(2)).sum::<f64>()
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::evaluate_binary_gini_split;

    #[test]
    fn assigns_zero_impurity_to_pure_branches() {
        let feature_values = NDArray::from_shape_vec([4], vec![0.0_f64, 1.0, 2.0, 3.0]).unwrap();
        let labels = NDArray::from_shape_vec([4], vec![0_usize, 0, 1, 1]).unwrap();

        let split = evaluate_binary_gini_split(&feature_values, &labels, 1.5).unwrap();

        assert_eq!(split.impurity(), 0.0);
        assert_eq!(split.left_count(), 2);
        assert_eq!(split.right_count(), 2);
    }

    #[test]
    fn assigns_maximum_impurity_to_balanced_branch_ties() {
        let feature_values = NDArray::from_shape_vec([4], vec![0.0_f64, 1.0, 2.0, 3.0]).unwrap();
        let labels = NDArray::from_shape_vec([4], vec![0_usize, 1, 0, 1]).unwrap();

        let split = evaluate_binary_gini_split(&feature_values, &labels, 1.5).unwrap();

        assert_close(split.impurity(), 0.5);
    }

    #[test]
    fn keeps_duplicate_feature_values_in_the_same_branch() {
        let feature_values = NDArray::from_shape_vec([4], vec![1.0_f64, 1.0, 2.0, 2.0]).unwrap();
        let labels = NDArray::from_shape_vec([4], vec![0_usize, 0, 1, 1]).unwrap();

        let split = evaluate_binary_gini_split(&feature_values, &labels, 1.0).unwrap();

        assert_eq!(split.impurity(), 0.0);
        assert_eq!(split.left_count(), 2);
        assert_eq!(split.right_count(), 2);
    }

    #[test]
    fn weights_impurity_by_imbalanced_branch_sizes() {
        let feature_values = NDArray::from_shape_vec([4], vec![0.0_f64, 1.0, 2.0, 3.0]).unwrap();
        let labels = NDArray::from_shape_vec([4], vec![0_usize, 0, 0, 1]).unwrap();

        let split = evaluate_binary_gini_split(&feature_values, &labels, 1.5).unwrap();

        assert_close(split.impurity(), 0.25);
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-12);
    }
}
