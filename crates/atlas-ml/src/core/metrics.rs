use atlas_ndarray::{ArrayElement, OperandMetadata};

use crate::{AtlasMlError, AtlasMlResult};

const ACCURACY_OP: &str = "classification_accuracy";

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

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::classification_accuracy;
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
}
