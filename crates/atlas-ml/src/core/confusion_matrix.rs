use std::collections::BTreeSet;

use atlas_ndarray::{NDArray, OperandMetadata};

use crate::{AtlasMlError, AtlasMlResult};

const OP: &str = "confusion_matrix";

/// A class-indexed confusion matrix with actual classes as rows and predicted classes as columns.
pub struct ConfusionMatrix {
    classes: Box<[usize]>,
    counts: NDArray<usize>,
}

impl ConfusionMatrix {
    /// Returns ascending labels for the rows and columns of [`Self::counts`].
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// Returns counts with actual classes as rows and predicted classes as columns.
    pub fn counts(&self) -> &NDArray<usize> {
        &self.counts
    }
}

/// Counts actual and predicted label pairs in a deterministic class order.
pub fn confusion_matrix<A, P>(actual: &A, predicted: &P) -> AtlasMlResult<ConfusionMatrix>
where
    A: OperandMetadata<usize> + ?Sized,
    P: OperandMetadata<usize> + ?Sized,
{
    validate_label_vectors(actual, predicted)?;

    let classes: Box<[usize]> = (0..actual.shape()[0])
        .flat_map(|index| [label(actual, index), label(predicted, index)])
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let count_len = classes
        .len()
        .checked_mul(classes.len())
        .ok_or(AtlasMlError::InvalidArgument { op: OP, reason: "class count is too large" })?;
    let mut counts = vec![0; count_len];
    for index in 0..actual.shape()[0] {
        let actual_index = classes
            .binary_search(&label(actual, index))
            .expect("observed actual labels are included in the confusion matrix classes");
        let predicted_index = classes
            .binary_search(&label(predicted, index))
            .expect("observed predicted labels are included in the confusion matrix classes");
        counts[actual_index * classes.len() + predicted_index] += 1;
    }

    Ok(ConfusionMatrix {
        counts: NDArray::from_shape_vec([classes.len(), classes.len()], counts)?,
        classes,
    })
}

fn validate_label_vectors<A, P>(actual: &A, predicted: &P) -> AtlasMlResult<()>
where
    A: OperandMetadata<usize> + ?Sized,
    P: OperandMetadata<usize> + ?Sized,
{
    if actual.ndim() != 1 {
        return Err(AtlasMlError::InvalidInputRank {
            op: OP,
            expected: "a rank-1 actual label vector",
            rank: actual.ndim(),
        });
    }
    if predicted.ndim() != 1 {
        return Err(AtlasMlError::InvalidInputRank {
            op: OP,
            expected: "a rank-1 predicted label vector",
            rank: predicted.ndim(),
        });
    }
    if actual.shape() != predicted.shape() {
        return Err(AtlasMlError::ShapeMismatch {
            op: OP,
            left: actual.shape().to_vec(),
            right: predicted.shape().to_vec(),
            reason: "label counts must match",
        });
    }
    if actual.shape()[0] == 0 {
        return Err(AtlasMlError::EmptyInput { op: OP });
    }

    Ok(())
}

fn label<O>(labels: &O, index: usize) -> usize
where
    O: OperandMetadata<usize> + ?Sized,
{
    labels.data()[labels.offset() + index * labels.strides()[0]]
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::confusion_matrix;
    use crate::AtlasMlError;

    #[test]
    fn uses_sorted_classes_and_retains_labels_absent_from_predictions() {
        let actual = NDArray::from_shape_vec([3], vec![7_usize, 2, 7]).unwrap();
        let predicted = NDArray::from_shape_vec([3], vec![2_usize, 2, 2]).unwrap();
        let matrix = confusion_matrix(&actual, &predicted).unwrap();

        assert_eq!(matrix.classes(), &[2, 7]);
        assert_eq!(matrix.counts().shape(), &[2, 2]);
        assert_eq!(matrix.counts().data(), &[1, 0, 2, 0]);
    }

    #[test]
    fn supports_logical_label_views() {
        let actual = NDArray::from_shape_vec([3], vec![0_usize, 1, 2]).unwrap();
        let predicted = NDArray::from_shape_vec([3], vec![0_usize, 0, 2]).unwrap();

        assert_eq!(
            confusion_matrix(&actual.view(), &predicted.view()).unwrap().counts().data(),
            &[1, 0, 0, 1, 0, 0, 0, 0, 1]
        );
    }

    #[test]
    fn rejects_empty_and_mismatched_label_vectors() {
        let empty = NDArray::<usize>::zeros([0]).unwrap();
        let actual = NDArray::from_shape_vec([2], vec![0_usize, 1]).unwrap();
        let predicted = NDArray::from_shape_vec([3], vec![0_usize, 1, 2]).unwrap();

        assert_eq!(
            confusion_matrix(&empty, &empty).map(|_| ()),
            Err(AtlasMlError::EmptyInput { op: "confusion_matrix" })
        );
        assert_eq!(
            confusion_matrix(&actual, &predicted).map(|_| ()),
            Err(AtlasMlError::ShapeMismatch {
                op: "confusion_matrix",
                left: vec![2],
                right: vec![3],
                reason: "label counts must match",
            })
        );
    }
}
