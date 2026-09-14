use atlas_ndarray::{ArrayElement, OperandMetadata};

use crate::{
    AtlasMlError, AtlasMlResult,
    core::validation::{validate_binary_labels, validate_finite_feature_values},
};

const ACCURACY_OP: &str = "classification_accuracy";
const MAE_OP: &str = "mean_absolute_error";
const MSE_OP: &str = "mean_squared_error";
const R_SQUARED_OP: &str = "coefficient_of_determination";
const BINARY_LOG_LOSS_OP: &str = "binary_log_loss";
const CLASSIFICATION_REPORT_OP: &str = "classification_report";

/// Binary classification metrics using label `1` as the positive class.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClassificationReport {
    accuracy: f64,
    precision: f64,
    recall: f64,
    f1_score: f64,
}

impl ClassificationReport {
    /// Returns the fraction of correct predictions.
    pub const fn accuracy(&self) -> f64 {
        self.accuracy
    }

    /// Returns positive-class precision, or `0.0` when there are no positive predictions.
    pub const fn precision(&self) -> f64 {
        self.precision
    }

    /// Returns positive-class recall, or `0.0` when there are no actual positive labels.
    pub const fn recall(&self) -> f64 {
        self.recall
    }

    /// Returns the harmonic mean of precision and recall, or `0.0` when both are zero.
    pub const fn f1_score(&self) -> f64 {
        self.f1_score
    }
}

/// Returns the fraction of matching labels in two rank-1 label vectors.
pub fn classification_accuracy<T, A, P>(actual: &A, predicted: &P) -> AtlasMlResult<f64>
where
    T: ArrayElement + PartialEq,
    A: OperandMetadata<T> + ?Sized,
    P: OperandMetadata<T> + ?Sized,
{
    validate_label_vectors(actual, predicted)?;

    let matches = (0..actual.shape()[0])
        .filter(|&index| label(actual, index) == label(predicted, index))
        .count();

    Ok(matches as f64 / actual.shape()[0] as f64)
}

/// Returns accuracy, precision, recall, and F1 for binary `0`/`1` label vectors.
pub fn classification_report<A, P>(actual: &A, predicted: &P) -> AtlasMlResult<ClassificationReport>
where
    A: OperandMetadata<usize> + ?Sized,
    P: OperandMetadata<usize> + ?Sized,
{
    let accuracy = classification_accuracy(actual, predicted)?;
    validate_binary_labels(actual, CLASSIFICATION_REPORT_OP)?;
    validate_binary_labels(predicted, CLASSIFICATION_REPORT_OP)?;

    let mut true_positives = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;
    for index in 0..actual.shape()[0] {
        match (label(actual, index), label(predicted, index)) {
            (1, 1) => true_positives += 1,
            (0, 1) => false_positives += 1,
            (1, 0) => false_negatives += 1,
            _ => {}
        }
    }

    let precision = ratio(true_positives, true_positives + false_positives);
    let recall = ratio(true_positives, true_positives + false_negatives);
    let f1_score = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    Ok(ClassificationReport { accuracy, precision, recall, f1_score })
}

/// Returns mean binary cross-entropy for `0`/`1` labels and positive-class probabilities.
///
/// Correct boundary probabilities have zero loss; impossible observed outcomes have infinite loss.
pub fn binary_log_loss<A, P>(actual: &A, probabilities: &P) -> AtlasMlResult<f64>
where
    A: OperandMetadata<usize> + ?Sized,
    P: OperandMetadata<f64> + ?Sized,
{
    validate_binary_probability_vectors(actual, probabilities)?;

    Ok((0..actual.shape()[0])
        .map(|index| {
            let probability = value(probabilities, index);
            if label(actual, index) == 0 { -(-probability).ln_1p() } else { -probability.ln() }
        })
        .sum::<f64>()
        / actual.shape()[0] as f64)
}

/// Returns the mean absolute error between two rank-1 target vectors.
pub fn mean_absolute_error<A, P>(actual: &A, predicted: &P) -> AtlasMlResult<f64>
where
    A: OperandMetadata<f64> + ?Sized,
    P: OperandMetadata<f64> + ?Sized,
{
    validate_regression_vectors(actual, predicted, MAE_OP)?;

    Ok((0..actual.shape()[0])
        .map(|index| (value(actual, index) - value(predicted, index)).abs())
        .sum::<f64>()
        / actual.shape()[0] as f64)
}

/// Returns the mean squared error between two rank-1 target vectors.
pub fn mean_squared_error<A, P>(actual: &A, predicted: &P) -> AtlasMlResult<f64>
where
    A: OperandMetadata<f64> + ?Sized,
    P: OperandMetadata<f64> + ?Sized,
{
    validate_regression_vectors(actual, predicted, MSE_OP)?;

    Ok((0..actual.shape()[0])
        .map(|index| {
            let error = value(actual, index) - value(predicted, index);
            error * error
        })
        .sum::<f64>()
        / actual.shape()[0] as f64)
}

/// Returns the coefficient of determination (R²) between two rank-1 target vectors.
pub fn coefficient_of_determination<A, P>(actual: &A, predicted: &P) -> AtlasMlResult<f64>
where
    A: OperandMetadata<f64> + ?Sized,
    P: OperandMetadata<f64> + ?Sized,
{
    validate_regression_vectors(actual, predicted, R_SQUARED_OP)?;

    let mean = (0..actual.shape()[0]).map(|index| value(actual, index)).sum::<f64>()
        / actual.shape()[0] as f64;
    let total_sum_squares = (0..actual.shape()[0])
        .map(|index| {
            let deviation = value(actual, index) - mean;
            deviation * deviation
        })
        .sum::<f64>();
    if total_sum_squares == 0.0 {
        return Err(AtlasMlError::InvalidArgument {
            op: R_SQUARED_OP,
            reason: "actual targets must have non-zero variance",
        });
    }

    let residual_sum_squares = (0..actual.shape()[0])
        .map(|index| {
            let residual = value(actual, index) - value(predicted, index);
            residual * residual
        })
        .sum::<f64>();

    Ok(1.0 - residual_sum_squares / total_sum_squares)
}

fn validate_label_vectors<T, A, P>(actual: &A, predicted: &P) -> AtlasMlResult<()>
where
    T: ArrayElement,
    A: OperandMetadata<T> + ?Sized,
    P: OperandMetadata<T> + ?Sized,
{
    if actual.ndim() != 1 {
        return Err(AtlasMlError::InvalidInputRank {
            op: ACCURACY_OP,
            expected: "a rank-1 actual label vector",
            rank: actual.ndim(),
        });
    }
    if predicted.ndim() != 1 {
        return Err(AtlasMlError::InvalidInputRank {
            op: ACCURACY_OP,
            expected: "a rank-1 predicted label vector",
            rank: predicted.ndim(),
        });
    }
    if actual.shape() != predicted.shape() {
        return Err(AtlasMlError::ShapeMismatch {
            op: ACCURACY_OP,
            left: actual.shape().to_vec(),
            right: predicted.shape().to_vec(),
            reason: "label counts must match",
        });
    }
    if actual.shape()[0] == 0 {
        return Err(AtlasMlError::EmptyInput { op: ACCURACY_OP });
    }

    Ok(())
}

fn validate_binary_probability_vectors<A, P>(actual: &A, probabilities: &P) -> AtlasMlResult<()>
where
    A: OperandMetadata<usize> + ?Sized,
    P: OperandMetadata<f64> + ?Sized,
{
    if actual.ndim() != 1 {
        return Err(AtlasMlError::InvalidInputRank {
            op: BINARY_LOG_LOSS_OP,
            expected: "a rank-1 actual label vector",
            rank: actual.ndim(),
        });
    }
    if probabilities.ndim() != 1 {
        return Err(AtlasMlError::InvalidInputRank {
            op: BINARY_LOG_LOSS_OP,
            expected: "a rank-1 probability vector",
            rank: probabilities.ndim(),
        });
    }
    if actual.shape() != probabilities.shape() {
        return Err(AtlasMlError::ShapeMismatch {
            op: BINARY_LOG_LOSS_OP,
            left: actual.shape().to_vec(),
            right: probabilities.shape().to_vec(),
            reason: "label and probability counts must match",
        });
    }
    if actual.shape()[0] == 0 {
        return Err(AtlasMlError::EmptyInput { op: BINARY_LOG_LOSS_OP });
    }

    validate_binary_labels(actual, BINARY_LOG_LOSS_OP)?;
    for index in 0..actual.shape()[0] {
        let probability = value(probabilities, index);
        if !probability.is_finite() {
            return Err(AtlasMlError::NonFiniteInput { op: BINARY_LOG_LOSS_OP });
        }
        if !(0.0..=1.0).contains(&probability) {
            return Err(AtlasMlError::InvalidArgument {
                op: BINARY_LOG_LOSS_OP,
                reason: "probabilities must be within [0, 1]",
            });
        }
    }

    Ok(())
}

fn label<T, O>(labels: &O, index: usize) -> T
where
    T: ArrayElement,
    O: OperandMetadata<T> + ?Sized,
{
    labels.data()[labels.offset() + index * labels.strides()[0]]
}

fn validate_regression_vectors<A, P>(
    actual: &A,
    predicted: &P,
    op: &'static str,
) -> AtlasMlResult<()>
where
    A: OperandMetadata<f64> + ?Sized,
    P: OperandMetadata<f64> + ?Sized,
{
    if actual.ndim() != 1 {
        return Err(AtlasMlError::InvalidInputRank {
            op,
            expected: "a rank-1 actual target vector",
            rank: actual.ndim(),
        });
    }
    if predicted.ndim() != 1 {
        return Err(AtlasMlError::InvalidInputRank {
            op,
            expected: "a rank-1 predicted target vector",
            rank: predicted.ndim(),
        });
    }
    if actual.shape() != predicted.shape() {
        return Err(AtlasMlError::ShapeMismatch {
            op,
            left: actual.shape().to_vec(),
            right: predicted.shape().to_vec(),
            reason: "target counts must match",
        });
    }
    if actual.shape()[0] == 0 {
        return Err(AtlasMlError::EmptyInput { op });
    }

    validate_finite_feature_values(actual, op)?;
    validate_finite_feature_values(predicted, op)
}

fn value<O>(values: &O, index: usize) -> f64
where
    O: OperandMetadata<f64> + ?Sized,
{
    values.data()[values.offset() + index * values.strides()[0]]
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 { 0.0 } else { numerator as f64 / denominator as f64 }
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::{
        binary_log_loss, classification_accuracy, classification_report,
        coefficient_of_determination, mean_absolute_error, mean_squared_error,
    };
    use crate::AtlasMlError;

    #[test]
    fn reports_exact_and_partial_label_matches() {
        let actual = NDArray::from_shape_vec([4], vec![0_usize, 1, 2, 3]).unwrap();
        let exact = NDArray::from_shape_vec([4], vec![0_usize, 1, 2, 3]).unwrap();
        let partial = NDArray::from_shape_vec([4], vec![0_usize, 0, 2, 0]).unwrap();

        assert_eq!(classification_accuracy(&actual, &exact), Ok(1.0));
        assert_eq!(classification_accuracy(&actual, &partial), Ok(0.5));
    }

    #[test]
    fn reports_known_binary_log_loss() {
        let actual = NDArray::from_shape_vec([2], vec![0_usize, 1]).unwrap();
        let probabilities = NDArray::from_shape_vec([2], vec![0.25_f64, 0.75]).unwrap();

        assert_close(binary_log_loss(&actual, &probabilities).unwrap(), -0.75_f64.ln());
    }

    #[test]
    fn handles_binary_log_loss_probability_boundaries() {
        let actual = NDArray::from_shape_vec([2], vec![0_usize, 1]).unwrap();
        let correct = NDArray::from_shape_vec([2], vec![0.0_f64, 1.0]).unwrap();
        let incorrect = NDArray::from_shape_vec([2], vec![1.0_f64, 0.0]).unwrap();

        assert_eq!(binary_log_loss(&actual, &correct), Ok(0.0));
        assert!(binary_log_loss(&actual, &incorrect).unwrap().is_infinite());
    }

    #[test]
    fn binary_log_loss_rejects_invalid_labels_and_probabilities() {
        let invalid_labels = NDArray::from_shape_vec([1], vec![2_usize]).unwrap();
        let finite_probability = NDArray::from_shape_vec([1], vec![0.5_f64]).unwrap();
        let valid_labels = NDArray::from_shape_vec([1], vec![1_usize]).unwrap();
        let non_finite_probability = NDArray::from_shape_vec([1], vec![f64::NAN]).unwrap();
        let out_of_range_probability = NDArray::from_shape_vec([1], vec![1.1_f64]).unwrap();

        assert_eq!(
            binary_log_loss(&invalid_labels, &finite_probability),
            Err(AtlasMlError::InvalidArgument {
                op: "binary_log_loss",
                reason: "labels must be binary values 0 or 1",
            })
        );
        assert_eq!(
            binary_log_loss(&valid_labels, &non_finite_probability),
            Err(AtlasMlError::NonFiniteInput { op: "binary_log_loss" })
        );
        assert_eq!(
            binary_log_loss(&valid_labels, &out_of_range_probability),
            Err(AtlasMlError::InvalidArgument {
                op: "binary_log_loss",
                reason: "probabilities must be within [0, 1]",
            })
        );
    }

    #[test]
    fn binary_log_loss_supports_logical_views() {
        let actual = NDArray::from_shape_vec([2], vec![0_usize, 1]).unwrap();
        let probabilities = NDArray::from_shape_vec([2], vec![0.25_f64, 0.75]).unwrap();

        assert_close(
            binary_log_loss(&actual.view(), &probabilities.view()).unwrap(),
            -0.75_f64.ln(),
        );
    }

    #[test]
    fn supports_label_views() {
        let actual = NDArray::from_shape_vec([3], vec![0_usize, 1, 2]).unwrap();
        let predicted = NDArray::from_shape_vec([3], vec![0_usize, 0, 2]).unwrap();

        assert_eq!(classification_accuracy(&actual.view(), &predicted.view()), Ok(2.0 / 3.0));
    }

    #[test]
    fn reports_binary_classification_metrics() {
        let actual = NDArray::from_shape_vec([4], vec![0_usize, 0, 1, 1]).unwrap();
        let predicted = NDArray::from_shape_vec([4], vec![0_usize, 1, 1, 1]).unwrap();

        let report = classification_report(&actual, &predicted).unwrap();

        assert_eq!(report.accuracy(), 0.75);
        assert_close(report.precision(), 2.0 / 3.0);
        assert_eq!(report.recall(), 1.0);
        assert_close(report.f1_score(), 0.8);
    }

    #[test]
    fn classification_report_rejects_non_binary_labels() {
        let actual = NDArray::from_shape_vec([2], vec![0_usize, 2]).unwrap();
        let predicted = NDArray::from_shape_vec([2], vec![0_usize, 1]).unwrap();

        assert_eq!(
            classification_report(&actual, &predicted),
            Err(AtlasMlError::InvalidArgument {
                op: "classification_report",
                reason: "labels must be binary values 0 or 1",
            })
        );
    }

    #[test]
    fn classification_report_handles_missing_positive_classes() {
        let actual = NDArray::from_shape_vec([2], vec![0_usize, 1]).unwrap();
        let predicted = NDArray::from_shape_vec([2], vec![0_usize, 0]).unwrap();

        let report = classification_report(&actual, &predicted).unwrap();

        assert_eq!(report.accuracy(), 0.5);
        assert_eq!(report.precision(), 0.0);
        assert_eq!(report.recall(), 0.0);
        assert_eq!(report.f1_score(), 0.0);
    }

    #[test]
    fn classification_report_supports_views_and_label_shape_errors() {
        let actual = NDArray::from_shape_vec([2], vec![0_usize, 1]).unwrap();
        let predicted = NDArray::from_shape_vec([2], vec![0_usize, 1]).unwrap();
        let mismatched = NDArray::from_shape_vec([1], vec![0_usize]).unwrap();

        assert_eq!(
            classification_report(&actual.view(), &predicted.view()).unwrap().accuracy(),
            1.0
        );
        assert_eq!(
            classification_report(&actual, &mismatched),
            Err(AtlasMlError::ShapeMismatch {
                op: "classification_accuracy",
                left: vec![2],
                right: vec![1],
                reason: "label counts must match",
            })
        );
    }

    #[test]
    fn rejects_empty_and_mismatched_label_vectors() {
        let empty = NDArray::<usize>::zeros([0]).unwrap();
        let actual = NDArray::from_shape_vec([2], vec![0_usize, 1]).unwrap();
        let predicted = NDArray::from_shape_vec([3], vec![0_usize, 1, 2]).unwrap();

        assert_eq!(
            classification_accuracy(&empty, &empty),
            Err(AtlasMlError::EmptyInput { op: "classification_accuracy" })
        );
        assert_eq!(
            classification_accuracy(&actual, &predicted),
            Err(AtlasMlError::ShapeMismatch {
                op: "classification_accuracy",
                left: vec![2],
                right: vec![3],
                reason: "label counts must match",
            })
        );
    }

    #[test]
    fn reports_exact_and_known_regression_errors() {
        let actual = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
        let exact = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
        let predicted = NDArray::from_shape_vec([3], vec![2.0_f64, 0.0, 5.0]).unwrap();

        assert_eq!(mean_absolute_error(&actual, &exact), Ok(0.0));
        assert_eq!(mean_squared_error(&actual, &exact), Ok(0.0));
        assert_eq!(mean_absolute_error(&actual, &predicted), Ok(5.0 / 3.0));
        assert_eq!(mean_squared_error(&actual, &predicted), Ok(3.0));
    }

    #[test]
    fn regression_metrics_support_logical_views() {
        let actual = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
        let predicted = NDArray::from_shape_vec([3], vec![2.0_f64, 0.0, 5.0]).unwrap();

        assert_eq!(mean_absolute_error(&actual.view(), &predicted.view()), Ok(5.0 / 3.0));
        assert_eq!(mean_squared_error(&actual.view(), &predicted.view()), Ok(3.0));
    }

    #[test]
    fn regression_metrics_reject_empty_non_finite_and_mismatched_targets() {
        let empty = NDArray::<f64>::zeros([0]).unwrap();
        let finite = NDArray::from_shape_vec([1], vec![1.0_f64]).unwrap();
        let non_finite = NDArray::from_shape_vec([1], vec![f64::NAN]).unwrap();
        let mismatched = NDArray::from_shape_vec([2], vec![1.0_f64, 2.0]).unwrap();

        assert_eq!(
            mean_absolute_error(&empty, &empty),
            Err(AtlasMlError::EmptyInput { op: "mean_absolute_error" })
        );
        assert_eq!(
            mean_absolute_error(&non_finite, &finite),
            Err(AtlasMlError::NonFiniteInput { op: "mean_absolute_error" })
        );
        assert_eq!(
            mean_squared_error(&finite, &non_finite),
            Err(AtlasMlError::NonFiniteInput { op: "mean_squared_error" })
        );
        assert_eq!(
            mean_squared_error(&finite, &mismatched),
            Err(AtlasMlError::ShapeMismatch {
                op: "mean_squared_error",
                left: vec![1],
                right: vec![2],
                reason: "target counts must match",
            })
        );
    }

    #[test]
    fn reports_perfect_and_known_coefficients_of_determination() {
        let actual = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
        let exact = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
        let predicted = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 1.0]).unwrap();

        assert_eq!(coefficient_of_determination(&actual, &exact), Ok(1.0));
        assert_eq!(coefficient_of_determination(&actual, &predicted), Ok(-1.0));
    }

    #[test]
    fn coefficient_of_determination_supports_views_and_rejects_constant_targets() {
        let actual = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
        let predicted = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 1.0]).unwrap();
        let constant = NDArray::from_shape_vec([2], vec![3.0_f64, 3.0]).unwrap();

        assert_eq!(coefficient_of_determination(&actual.view(), &predicted.view()), Ok(-1.0));
        assert_eq!(
            coefficient_of_determination(&constant, &constant),
            Err(AtlasMlError::InvalidArgument {
                op: "coefficient_of_determination",
                reason: "actual targets must have non-zero variance",
            })
        );
    }

    #[test]
    fn coefficient_of_determination_rejects_invalid_targets() {
        let empty = NDArray::<f64>::zeros([0]).unwrap();
        let finite = NDArray::from_shape_vec([1], vec![1.0_f64]).unwrap();
        let non_finite = NDArray::from_shape_vec([1], vec![f64::NAN]).unwrap();
        let mismatched = NDArray::from_shape_vec([2], vec![1.0_f64, 2.0]).unwrap();

        assert_eq!(
            coefficient_of_determination(&empty, &empty),
            Err(AtlasMlError::EmptyInput { op: "coefficient_of_determination" })
        );
        assert_eq!(
            coefficient_of_determination(&non_finite, &finite),
            Err(AtlasMlError::NonFiniteInput { op: "coefficient_of_determination" })
        );
        assert_eq!(
            coefficient_of_determination(&finite, &mismatched),
            Err(AtlasMlError::ShapeMismatch {
                op: "coefficient_of_determination",
                left: vec![1],
                right: vec![2],
                reason: "target counts must match",
            })
        );
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-12);
    }
}
