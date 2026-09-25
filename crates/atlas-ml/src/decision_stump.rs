use std::collections::BTreeMap;

use atlas_ndarray::{NDArray, OperandMetadata};

use crate::{
    AtlasMlError, AtlasMlResult,
    core::validation::{
        validate_finite_feature_values, validate_prediction_feature_inputs,
        validate_supervised_training_inputs,
    },
};

const FIT_OP: &str = "decision_stump_fit";
const PREDICT_OP: &str = "decision_stump_predict";

/// A one-feature threshold classifier with deterministic branch-label ties.
pub struct DecisionStumpClassifier {
    feature_index: usize,
    feature_count: usize,
    threshold: f64,
    left_label: usize,
    right_label: usize,
}

impl DecisionStumpClassifier {
    /// Fits the feature threshold with the lowest training classification error.
    pub fn fit<F, L>(features: &F, labels: &L) -> AtlasMlResult<Self>
    where
        F: OperandMetadata<f64> + ?Sized,
        L: OperandMetadata<usize> + ?Sized,
    {
        validate_supervised_training_inputs(features, labels, FIT_OP)?;
        validate_finite_feature_values(features, FIT_OP)?;
        if features.shape()[1] == 0 {
            return Err(AtlasMlError::InvalidArgument {
                op: FIT_OP,
                reason: "features must contain at least one feature",
            });
        }

        let sample_count = features.shape()[0];
        let labels = label_values(labels);
        let default_label = majority_label(&label_counts(&labels))
            .expect("nonempty supervised training input has at least one label");
        let mut best = None;

        for feature_index in 0..features.shape()[1] {
            let mut samples = (0..sample_count)
                .map(|sample_index| {
                    (feature(features, sample_index, feature_index), labels[sample_index])
                })
                .collect::<Vec<_>>();
            samples.sort_unstable_by(|(left, _), (right, _)| left.total_cmp(right));

            let mut left_counts = BTreeMap::new();
            let mut right_counts = label_counts(&labels);
            let mut left_count = 0;
            let mut right_count = sample_count;
            let mut group_end = 0;
            while group_end < sample_count {
                let value = samples[group_end].0;
                while group_end < sample_count && samples[group_end].0 == value {
                    let label = samples[group_end].1;
                    *left_counts.entry(label).or_insert(0) += 1;
                    decrement_count(&mut right_counts, label);
                    left_count += 1;
                    right_count -= 1;
                    group_end += 1;
                }
                if group_end == sample_count {
                    continue;
                }

                let next_value = samples[group_end].0;
                let left_label = majority_label(&left_counts)
                    .expect("a threshold boundary has a nonempty left branch");
                let right_label = majority_label(&right_counts)
                    .expect("a threshold boundary has a nonempty right branch");
                let errors = left_count - left_counts[&left_label] + right_count
                    - right_counts[&right_label];
                let candidate = Candidate {
                    feature_index,
                    threshold: midpoint(value, next_value),
                    left_label,
                    right_label,
                    errors,
                };
                if best.as_ref().is_none_or(|best: &Candidate| candidate.errors < best.errors) {
                    best = Some(candidate);
                }
            }
        }

        let best = best.unwrap_or(Candidate {
            feature_index: 0,
            threshold: feature(features, 0, 0),
            left_label: default_label,
            right_label: default_label,
            errors: 0,
        });
        Ok(Self {
            feature_index: best.feature_index,
            feature_count: features.shape()[1],
            threshold: best.threshold,
            left_label: best.left_label,
            right_label: best.right_label,
        })
    }

    /// Returns the selected feature column.
    pub const fn feature_index(&self) -> usize {
        self.feature_index
    }

    /// Returns the number of input features expected by this model.
    pub const fn feature_count(&self) -> usize {
        self.feature_count
    }

    /// Returns the selected split threshold; values less than or equal to it use the left label.
    pub const fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Returns the predicted label for values less than or equal to [`Self::threshold`].
    pub const fn left_label(&self) -> usize {
        self.left_label
    }

    /// Returns the predicted label for values greater than [`Self::threshold`].
    pub const fn right_label(&self) -> usize {
        self.right_label
    }

    /// Predicts one class label for every query row.
    pub fn predict<Q>(&self, queries: &Q) -> AtlasMlResult<NDArray<usize>>
    where
        Q: OperandMetadata<f64> + ?Sized,
    {
        validate_prediction_feature_inputs(queries, self.feature_count, PREDICT_OP)?;
        validate_finite_feature_values(queries, PREDICT_OP)?;

        let predictions = (0..queries.shape()[0])
            .map(|query_index| {
                if feature(queries, query_index, self.feature_index) <= self.threshold {
                    self.left_label
                } else {
                    self.right_label
                }
            })
            .collect();
        Ok(NDArray::from_shape_vec([queries.shape()[0]], predictions)?)
    }
}

struct Candidate {
    feature_index: usize,
    threshold: f64,
    left_label: usize,
    right_label: usize,
    errors: usize,
}

fn feature<F>(features: &F, sample_index: usize, feature_index: usize) -> f64
where
    F: OperandMetadata<f64> + ?Sized,
{
    features.data()[features.offset()
        + sample_index * features.strides()[0]
        + feature_index * features.strides()[1]]
}

fn label_values<L>(labels: &L) -> Vec<usize>
where
    L: OperandMetadata<usize> + ?Sized,
{
    (0..labels.shape()[0])
        .map(|index| labels.data()[labels.offset() + index * labels.strides()[0]])
        .collect()
}

fn label_counts(labels: &[usize]) -> BTreeMap<usize, usize> {
    let mut counts = BTreeMap::new();
    for &label in labels {
        *counts.entry(label).or_insert(0) += 1;
    }
    counts
}

fn decrement_count(counts: &mut BTreeMap<usize, usize>, label: usize) {
    let count = counts.get_mut(&label).expect("observed label must have a count");
    *count -= 1;
    if *count == 0 {
        counts.remove(&label);
    }
}

fn majority_label(counts: &BTreeMap<usize, usize>) -> Option<usize> {
    counts
        .iter()
        .max_by(|(left_label, left_count), (right_label, right_count)| {
            left_count.cmp(right_count).then_with(|| right_label.cmp(left_label))
        })
        .map(|(&label, _)| label)
}

fn midpoint(left: f64, right: f64) -> f64 {
    left * 0.5 + right * 0.5
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::DecisionStumpClassifier;

    #[test]
    fn selects_the_lowest_error_threshold() {
        let features = NDArray::from_shape_vec([4, 1], vec![0.0_f64, 1.0, 2.0, 3.0]).unwrap();
        let labels = NDArray::from_shape_vec([4], vec![0_usize, 0, 1, 1]).unwrap();

        let model = DecisionStumpClassifier::fit(&features, &labels).unwrap();

        assert_eq!(model.feature_index(), 0);
        assert_eq!(model.threshold(), 1.5);
        assert_eq!(model.left_label(), 0);
        assert_eq!(model.right_label(), 1);
        assert_eq!(model.predict(&features).unwrap().data(), labels.data());
    }

    #[test]
    fn resolves_equal_feature_splits_by_lower_feature_index() {
        let features =
            NDArray::from_shape_vec([4, 2], vec![0.0_f64, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
                .unwrap();
        let labels = NDArray::from_shape_vec([4], vec![0_usize, 0, 1, 1]).unwrap();

        let model = DecisionStumpClassifier::fit(&features, &labels).unwrap();

        assert_eq!(model.feature_index(), 0);
        assert_eq!(model.threshold(), 1.5);
    }

    #[test]
    fn resolves_constant_feature_label_ties_by_lower_label() {
        let features = NDArray::from_shape_vec([2, 1], vec![2.0_f64, 2.0]).unwrap();
        let labels = NDArray::from_shape_vec([2], vec![4_usize, 1]).unwrap();

        let model = DecisionStumpClassifier::fit(&features, &labels).unwrap();

        assert_eq!(model.threshold(), 2.0);
        assert_eq!(model.left_label(), 1);
        assert_eq!(model.right_label(), 1);
        assert_eq!(model.predict(&features).unwrap().data(), &[1, 1]);
    }

    #[test]
    fn fits_and_predicts_logical_views() {
        let source =
            NDArray::from_shape_vec([2, 4], vec![0.0_f64, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0])
                .unwrap();
        let labels = NDArray::from_shape_vec([1, 4], vec![0_usize, 0, 1, 1]).unwrap();
        let features = source.view().transpose();
        let labels = labels.view().reshape([4]).unwrap();

        let model = DecisionStumpClassifier::fit(&features, &labels).unwrap();

        assert_eq!(model.predict(&features).unwrap().data(), &[0, 0, 1, 1]);
    }
}
