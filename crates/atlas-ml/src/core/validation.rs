use atlas_ndarray::{ArrayElement, OperandMetadata, checked_element_count};

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

pub(crate) fn validate_prediction_feature_inputs<F, T>(
    features: &F,
    expected_feature_count: usize,
    op: &'static str,
) -> AtlasMlResult<()>
where
    F: OperandMetadata<T> + ?Sized,
    T: ArrayElement,
{
    if features.ndim() != 2 {
        return Err(AtlasMlError::InvalidInputRank {
            op,
            expected: "a rank-2 [queries, features] matrix",
            rank: features.ndim(),
        });
    }
    if features.shape()[1] != expected_feature_count {
        return Err(AtlasMlError::ShapeMismatch {
            op,
            left: features.shape().to_vec(),
            right: vec![expected_feature_count],
            reason: "feature count must match training data",
        });
    }

    Ok(())
}

pub(crate) fn validate_prediction_feature_row(
    features: &[f64],
    expected_feature_count: usize,
    op: &'static str,
) -> AtlasMlResult<()> {
    if features.len() != expected_feature_count {
        return Err(AtlasMlError::ShapeMismatch {
            op,
            left: vec![features.len()],
            right: vec![expected_feature_count],
            reason: "feature count must match training data",
        });
    }

    validate_finite_values(features, op)
}

pub(crate) fn validate_finite_feature_values<F>(features: &F, op: &'static str) -> AtlasMlResult<()>
where
    F: OperandMetadata<f64> + ?Sized,
{
    if let Some(values) = features.dense_slice() {
        return validate_finite_values(values, op);
    }

    let shape = features.shape();
    for linear_index in 0..checked_element_count(shape)? {
        let mut remainder = linear_index;
        let mut offset = features.offset();

        for axis in (0..shape.len()).rev() {
            let coordinate = remainder % shape[axis];
            remainder /= shape[axis];
            offset += coordinate * features.strides()[axis];
        }

        if !features.data()[offset].is_finite() {
            return Err(AtlasMlError::NonFiniteInput { op });
        }
    }

    Ok(())
}

pub(crate) fn validate_finite_target_values(
    targets: &[f64],
    op: &'static str,
) -> AtlasMlResult<()> {
    validate_finite_values(targets, op)
}

fn validate_finite_values(values: &[f64], op: &'static str) -> AtlasMlResult<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(AtlasMlError::NonFiniteInput { op })
    }
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::{
        validate_finite_feature_values, validate_prediction_feature_inputs,
        validate_supervised_training_inputs,
    };
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

    #[test]
    fn accepts_matching_and_empty_query_batches() {
        let queries = NDArray::<f64>::zeros([2, 3]).unwrap();
        let empty_queries = NDArray::<f64>::zeros([0, 3]).unwrap();

        assert_eq!(validate_prediction_feature_inputs(&queries, 3, "knn_predict"), Ok(()));
        assert_eq!(validate_prediction_feature_inputs(&empty_queries, 3, "knn_predict"), Ok(()));
    }

    #[test]
    fn rejects_invalid_query_rank_and_feature_width() {
        let vector = NDArray::from_shape_vec([3], vec![0.0_f64; 3]).unwrap();
        let queries = NDArray::<f64>::zeros([2, 2]).unwrap();

        assert_eq!(
            validate_prediction_feature_inputs(&vector, 3, "knn_predict"),
            Err(AtlasMlError::InvalidInputRank {
                op: "knn_predict",
                expected: "a rank-2 [queries, features] matrix",
                rank: 1,
            })
        );
        assert_eq!(
            validate_prediction_feature_inputs(&queries, 3, "knn_predict"),
            Err(AtlasMlError::ShapeMismatch {
                op: "knn_predict",
                left: vec![2, 2],
                right: vec![3],
                reason: "feature count must match training data",
            })
        );
    }

    #[test]
    fn validates_finite_feature_values() {
        let finite = NDArray::from_shape_vec([2, 2], vec![-1.0_f64, 0.0, 1.0, 2.0]).unwrap();
        let nan = NDArray::from_shape_vec([1, 1], vec![f64::NAN]).unwrap();
        let positive_infinity = NDArray::from_shape_vec([1, 1], vec![f64::INFINITY]).unwrap();
        let negative_infinity = NDArray::from_shape_vec([1, 1], vec![f64::NEG_INFINITY]).unwrap();

        assert_eq!(validate_finite_feature_values(&finite, OP), Ok(()));
        assert_eq!(
            validate_finite_feature_values(&nan, OP),
            Err(AtlasMlError::NonFiniteInput { op: OP })
        );
        assert_eq!(
            validate_finite_feature_values(&positive_infinity, OP),
            Err(AtlasMlError::NonFiniteInput { op: OP })
        );
        assert_eq!(
            validate_finite_feature_values(&negative_infinity, OP),
            Err(AtlasMlError::NonFiniteInput { op: OP })
        );
    }
}
