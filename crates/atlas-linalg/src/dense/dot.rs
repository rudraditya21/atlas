use atlas_ndarray::{NDArray, Numeric};

use super::matmul::matmul;
use crate::{
    core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand},
    internal::dense::{dot_kernel, vector_ref},
};

#[derive(Clone, Debug)]
pub enum DotOutput<T: Numeric> {
    Scalar(T),
    Array(NDArray<T>),
}

impl<T> PartialEq<T> for DotOutput<T>
where
    T: Numeric + PartialEq,
{
    fn eq(&self, other: &T) -> bool {
        match self {
            Self::Scalar(value) => value == other,
            Self::Array(_) => false,
        }
    }
}

pub fn dot<'a, T, L, R>(lhs: L, rhs: R) -> AtlasLinalgResult<DotOutput<T>>
where
    T: Numeric + 'a,
    L: Into<LinalgOperand<'a, T>>,
    R: Into<LinalgOperand<'a, T>>,
{
    let lhs = lhs.into();
    let rhs = rhs.into();

    match (lhs.shape(), rhs.shape()) {
        ([lhs_len], [rhs_len]) if lhs_len == rhs_len => {
            let lhs = vector_ref(&lhs);
            let rhs = vector_ref(&rhs);

            Ok(DotOutput::Scalar(dot_kernel(lhs, rhs)))
        }
        ([..], [..]) if lhs.ndim() == 1 && rhs.ndim() == 1 => {
            Err(AtlasLinalgError::ShapeMismatch {
                op: "dot",
                left: lhs.shape().to_vec(),
                right: rhs.shape().to_vec(),
                reason: "vector lengths must match",
            })
        }
        ([..], [..]) if lhs.ndim() <= 2 && rhs.ndim() <= 2 => {
            matmul(lhs, rhs).map(DotOutput::Array).map_err(remap_matmul_error_for_dot)
        }
        _ => Err(AtlasLinalgError::InvalidOperandRank {
            op: "dot",
            left: lhs.ndim(),
            right: rhs.ndim(),
        }),
    }
}

fn remap_matmul_error_for_dot(error: AtlasLinalgError) -> AtlasLinalgError {
    match error {
        AtlasLinalgError::InvalidOperandRank { left, right, .. } => {
            AtlasLinalgError::InvalidOperandRank { op: "dot", left, right }
        }
        AtlasLinalgError::ShapeMismatch { left, right, reason, .. } => {
            AtlasLinalgError::ShapeMismatch { op: "dot", left, right, reason }
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::{DotOutput, dot};
    use crate::{
        AtlasLinalgError, LinalgOperand,
        internal::dense::{VectorRef, dot_contiguous, dot_kernel, dot_strided, vector_ref},
    };

    #[test]
    fn dot_rejects_ranks_above_two_and_mismatched_vector_lengths() {
        let lhs = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
        let rhs = NDArray::from_shape_vec([2], vec![4_i32, 5]).unwrap();
        let rank_three = NDArray::<i32>::zeros([1, 1, 3]).unwrap();

        assert_eq!(
            dot(&lhs, &rhs).unwrap_err(),
            AtlasLinalgError::ShapeMismatch {
                op: "dot",
                left: vec![3],
                right: vec![2],
                reason: "vector lengths must match",
            }
        );
        assert_eq!(
            dot(&lhs, &rank_three).unwrap_err(),
            AtlasLinalgError::InvalidOperandRank { op: "dot", left: 1, right: 3 }
        );
    }

    #[test]
    fn dot_and_matmul_match_for_owned_and_view_vector_operands() {
        let lhs = NDArray::from_shape_vec([4], vec![1_i32, 2, 3, 4]).unwrap();
        let rhs = NDArray::from_shape_vec([4], vec![5_i32, 6, 7, 8]).unwrap();
        let lhs_view = lhs.view().slice([0], [4]).unwrap();
        let rhs_view = rhs.view().slice([0], [4]).unwrap();

        match (dot(&lhs, &rhs).unwrap(), dot(lhs_view.clone(), rhs_view.clone()).unwrap()) {
            (DotOutput::Scalar(lhs), DotOutput::Scalar(rhs)) => assert_eq!(lhs, rhs),
            (left, right) => panic!("expected scalar dot outputs, got {left:?} and {right:?}"),
        }
    }

    #[test]
    fn scalar_dot_outputs_remain_directly_comparable_with_scalars() {
        let lhs = NDArray::from_shape_vec([0], Vec::<f64>::new()).unwrap();
        let rhs = NDArray::from_shape_vec([0], Vec::<f64>::new()).unwrap();

        assert_eq!(dot(&lhs, &rhs).unwrap(), 0.0);
    }

    #[test]
    fn dot_accepts_vector_matrix_matrix_vector_and_matrix_matrix_operands() {
        let vector = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
        let right_matrix = NDArray::from_shape_vec([3, 2], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let left_matrix = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let matrix_rhs = NDArray::from_shape_vec([3, 2], vec![7_i32, 8, 9, 10, 11, 12]).unwrap();

        match dot(&vector, &right_matrix).unwrap() {
            DotOutput::Array(result) => {
                assert_eq!(result.shape(), &[2]);
                assert_eq!(result.data(), &[22, 28]);
            }
            other => panic!("expected array output for vector-matrix dot, got {other:?}"),
        }

        match dot(&left_matrix, &vector).unwrap() {
            DotOutput::Array(result) => {
                assert_eq!(result.shape(), &[2]);
                assert_eq!(result.data(), &[14, 32]);
            }
            other => panic!("expected array output for matrix-vector dot, got {other:?}"),
        }

        match dot(&left_matrix, &matrix_rhs).unwrap() {
            DotOutput::Array(result) => {
                assert_eq!(result.shape(), &[2, 2]);
                assert_eq!(result.data(), &[58, 64, 139, 154]);
            }
            other => panic!("expected array output for matrix-matrix dot, got {other:?}"),
        }
    }

    #[test]
    fn dot_reports_numpy_style_shape_mismatches_for_supported_ranks() {
        let vector = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
        let short_vector = NDArray::from_shape_vec([2], vec![4_i32, 5]).unwrap();
        let matrix = NDArray::from_shape_vec([2, 2], vec![1_i32, 2, 3, 4]).unwrap();
        let wide_matrix = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let tall_matrix = NDArray::from_shape_vec([3, 2], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();

        assert_eq!(
            dot(&vector, &short_vector).unwrap_err(),
            AtlasLinalgError::ShapeMismatch {
                op: "dot",
                left: vec![3],
                right: vec![2],
                reason: "vector lengths must match",
            }
        );
        assert_eq!(
            dot(&vector, &matrix).unwrap_err(),
            AtlasLinalgError::ShapeMismatch {
                op: "dot",
                left: vec![3],
                right: vec![2, 2],
                reason: "vector length must match matrix row count",
            }
        );
        assert_eq!(
            dot(&wide_matrix, &short_vector).unwrap_err(),
            AtlasLinalgError::ShapeMismatch {
                op: "dot",
                left: vec![2, 3],
                right: vec![2],
                reason: "matrix column count must match vector length",
            }
        );
        assert_eq!(
            dot(&matrix, &tall_matrix).unwrap_err(),
            AtlasLinalgError::ShapeMismatch {
                op: "dot",
                left: vec![2, 2],
                right: vec![3, 2],
                reason: "left matrix column count must match right matrix row count",
            }
        );
    }

    #[test]
    fn dot_fast_and_strided_kernels_are_equivalent() {
        let lhs_contiguous = [1_i32, 2, 3, 4];
        let rhs_contiguous = [5_i32, 6, 7, 8];
        let lhs_strided_data = [1_i32, -1, 2, -1, 3, -1, 4];
        let rhs_strided_data = [5_i32, -1, 6, -1, 7, -1, 8];

        let lhs_strided =
            VectorRef { data: &lhs_strided_data, offset: 0, len: lhs_contiguous.len(), stride: 2 };
        let rhs_strided =
            VectorRef { data: &rhs_strided_data, offset: 0, len: rhs_contiguous.len(), stride: 2 };

        let expected = dot_contiguous(&lhs_contiguous, &rhs_contiguous);

        assert_eq!(expected, dot_strided(lhs_strided, rhs_strided));
        assert_eq!(expected, dot_kernel(lhs_strided, rhs_strided));
    }

    #[test]
    fn vector_refs_preserve_owned_and_view_layouts() {
        let owned = NDArray::from_shape_vec([4], vec![1_i32, 2, 3, 4]).unwrap();
        let viewed = owned.view().slice([0], [4]).unwrap();
        let owned_operand = LinalgOperand::from(&owned);
        let view_operand = LinalgOperand::from(viewed);

        let owned_ref = vector_ref(&owned_operand);
        let view_ref = vector_ref(&view_operand);

        assert!(owned_ref.is_contiguous());
        assert!(view_ref.is_contiguous());
        assert_eq!(owned_ref.len, view_ref.len);
    }
}
