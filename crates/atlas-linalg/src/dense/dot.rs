use atlas_ndarray::Numeric;

use crate::core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand};

use super::{VectorRef, vector_ref};

pub fn dot<'a, T, L, R>(lhs: L, rhs: R) -> AtlasLinalgResult<T>
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

            Ok(dot_kernel(lhs, rhs))
        }
        ([..], [..]) if lhs.ndim() == 1 && rhs.ndim() == 1 => {
            Err(AtlasLinalgError::ShapeMismatch {
                op: "dot",
                left: lhs.shape().to_vec(),
                right: rhs.shape().to_vec(),
                reason: "vector lengths must match",
            })
        }
        _ => Err(AtlasLinalgError::InvalidOperandRank {
            op: "dot",
            left: lhs.ndim(),
            right: rhs.ndim(),
        }),
    }
}

pub(super) fn dot_kernel<T: Numeric>(lhs: VectorRef<'_, T>, rhs: VectorRef<'_, T>) -> T {
    if lhs.is_contiguous() && rhs.is_contiguous() {
        dot_contiguous(lhs.contiguous_slice(), rhs.contiguous_slice())
    } else {
        dot_strided(lhs, rhs)
    }
}

pub(super) fn dot_contiguous<T: Numeric>(lhs: &[T], rhs: &[T]) -> T {
    let mut total = T::zero();

    for index in 0..lhs.len() {
        total += lhs[index] * rhs[index];
    }

    total
}

fn dot_strided<T: Numeric>(lhs: VectorRef<'_, T>, rhs: VectorRef<'_, T>) -> T {
    let mut total = T::zero();

    for index in 0..lhs.len {
        total += lhs.value_at(index) * rhs.value_at(index);
    }

    total
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::{dot, dot_contiguous, dot_kernel, dot_strided};
    use crate::{AtlasLinalgError, LinalgOperand};

    use crate::dense::{VectorRef, vector_ref};

    #[test]
    fn dot_rejects_non_vector_inputs_and_mismatched_lengths() {
        let lhs = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
        let rhs = NDArray::from_shape_vec([2], vec![4_i32, 5]).unwrap();
        let matrix = NDArray::from_shape_vec([1, 3], vec![1_i32, 2, 3]).unwrap();

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
            dot(&lhs, &matrix).unwrap_err(),
            AtlasLinalgError::InvalidOperandRank { op: "dot", left: 1, right: 2 }
        );
    }

    #[test]
    fn dot_and_matmul_match_for_owned_and_view_vector_operands() {
        let lhs = NDArray::from_shape_vec([4], vec![1_i32, 2, 3, 4]).unwrap();
        let rhs = NDArray::from_shape_vec([4], vec![5_i32, 6, 7, 8]).unwrap();
        let lhs_view = lhs.view().slice([0], [4]).unwrap();
        let rhs_view = rhs.view().slice([0], [4]).unwrap();

        assert_eq!(dot(&lhs, &rhs).unwrap(), dot(lhs_view.clone(), rhs_view.clone()).unwrap());
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
