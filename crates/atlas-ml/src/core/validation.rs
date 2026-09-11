use atlas_ndarray::{ArrayElement, OperandMetadata};

use super::error::{AtlasMlError, AtlasMlResult};

pub(crate) fn validate_supervised_training_inputs<F, T, X, Y>(
    features: &F,
    targets: &T,
    op: &'static str,
) -> AtlasMlResult<()>
where
    F: OperandMetadata<X> + ?Sized,
    T: OperandMetadata<Y> + ?Sized,
    X: ArrayElement,
    Y: ArrayElement,
{
    if features.ndim() != 2 {
        return Err(AtlasMlError::InvalidInputRank {
            op,
            expected: "a rank-2 [samples, features] matrix",
            rank: features.ndim(),
        });
    }
    if targets.ndim() != 1 {
        return Err(AtlasMlError::InvalidInputRank {
            op,
            expected: "a rank-1 target vector",
            rank: targets.ndim(),
        });
    }
    if features.shape()[0] == 0 || targets.shape()[0] == 0 {
        return Err(AtlasMlError::EmptyInput { op });
    }
    if features.shape()[0] != targets.shape()[0] {
        return Err(AtlasMlError::ShapeMismatch {
            op,
            left: features.shape().to_vec(),
            right: targets.shape().to_vec(),
            reason: "sample counts must match",
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::validate_supervised_training_inputs;
    use crate::AtlasMlError;

    const OP: &str = "knn_fit";

    #[test]
    fn accepts_matching_nonempty_feature_and_target_samples() {
        let features = NDArray::from_shape_vec([2, 3], vec![0.0_f64; 6]).unwrap();
        let targets = NDArray::from_shape_vec([2], vec![0_usize, 1]).unwrap();

        assert_eq!(validate_supervised_training_inputs(&features, &targets, OP), Ok(()));
    }

    #[test]
    fn rejects_empty_samples() {
        let empty_features = NDArray::<f64>::zeros([0, 3]).unwrap();
        let empty_targets = NDArray::from_shape_vec([0], Vec::<usize>::new()).unwrap();
        let features = NDArray::<f64>::zeros([2, 3]).unwrap();

        assert_eq!(
            validate_supervised_training_inputs(&empty_features, &empty_targets, OP),
            Err(AtlasMlError::EmptyInput { op: OP })
        );
        assert_eq!(
            validate_supervised_training_inputs(&features, &empty_targets, OP),
            Err(AtlasMlError::EmptyInput { op: OP })
        );
    }

    #[test]
    fn rejects_invalid_feature_and_target_ranks() {
        let features = NDArray::from_shape_vec([2], vec![0.0_f64, 1.0]).unwrap();
        let targets = NDArray::from_shape_vec([2], vec![0_usize, 1]).unwrap();
        let matrix_targets = NDArray::from_shape_vec([1, 2], vec![0_usize, 1]).unwrap();
        let matrix_features = NDArray::<f64>::zeros([2, 3]).unwrap();

        assert_eq!(
            validate_supervised_training_inputs(&features, &targets, OP),
            Err(AtlasMlError::InvalidInputRank {
                op: OP,
                expected: "a rank-2 [samples, features] matrix",
                rank: 1,
            })
        );
        assert_eq!(
            validate_supervised_training_inputs(&matrix_features, &matrix_targets, OP),
            Err(AtlasMlError::InvalidInputRank {
                op: OP,
                expected: "a rank-1 target vector",
                rank: 2,
            })
        );
    }

    #[test]
    fn rejects_mismatched_sample_counts() {
        let features = NDArray::<f64>::zeros([3, 2]).unwrap();
        let targets = NDArray::from_shape_vec([2], vec![0_usize, 1]).unwrap();

        assert_eq!(
            validate_supervised_training_inputs(&features, &targets, OP),
            Err(AtlasMlError::ShapeMismatch {
                op: OP,
                left: vec![3, 2],
                right: vec![2],
                reason: "sample counts must match",
            })
        );
    }
}
