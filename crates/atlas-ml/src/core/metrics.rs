use atlas_ndarray::{ArrayElement, OperandMetadata};

use crate::{AtlasMlError, AtlasMlResult, core::validation::validate_finite_feature_values};

const ACCURACY_OP: &str = "classification_accuracy";
const MAE_OP: &str = "mean_absolute_error";
const MSE_OP: &str = "mean_squared_error";

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

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::{classification_accuracy, mean_absolute_error, mean_squared_error};
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
    fn supports_label_views() {
        let actual = NDArray::from_shape_vec([3], vec![0_usize, 1, 2]).unwrap();
        let predicted = NDArray::from_shape_vec([3], vec![0_usize, 0, 2]).unwrap();

        assert_eq!(classification_accuracy(&actual.view(), &predicted.view()), Ok(2.0 / 3.0));
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
}
