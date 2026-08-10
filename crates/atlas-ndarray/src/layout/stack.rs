use crate::{
    ArrayElement, AtlasNdError, AtlasNdResult, AxisIndex, NDArray,
    core::axis::normalize_insertion_axis, view::ArrayView,
};

impl<T: ArrayElement> NDArray<T> {
    pub fn stack<A: AxisIndex>(arrays: &[ArrayView<'_, T>], axis: A) -> AtlasNdResult<Self> {
        let Some(first) = arrays.first() else {
            return Err(AtlasNdError::InvalidArgument {
                op: "stack",
                reason: "at least one array is required",
            });
        };

        let ndim = first.ndim();
        let axis = normalize_insertion_axis(axis, ndim)?;

        for array in &arrays[1..] {
            if array.ndim() != ndim {
                return Err(AtlasNdError::DimensionMismatch {
                    expected: ndim,
                    actual: array.ndim(),
                });
            }

            if array.shape() != first.shape() {
                return Err(AtlasNdError::InvalidArgument {
                    op: "stack",
                    reason: "all input shapes must match",
                });
            }
        }

        let expanded: AtlasNdResult<Vec<_>> =
            arrays.iter().map(|array| array.clone().expand_dims(axis)).collect();

        NDArray::concatenate(&expanded?, axis)
    }
}

#[cfg(test)]
mod tests {
    use crate::{AtlasNdError, NDArray};

    #[test]
    fn stack_inserts_a_new_leading_axis_for_contiguous_arrays() {
        let lhs = NDArray::from_shape_vec([2, 2], vec![0_i32, 1, 2, 3]).unwrap();
        let rhs = NDArray::from_shape_vec([2, 2], vec![4_i32, 5, 6, 7]).unwrap();
        let stacked = NDArray::stack(&[lhs.view(), rhs.view()], 0).unwrap();

        assert_eq!(stacked.shape(), &[2, 2, 2]);
        assert_eq!(stacked.strides(), &[4, 2, 1]);
        assert_eq!(stacked.data(), &[0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn stack_inserts_a_new_inner_axis_with_row_major_output_order() {
        let lhs = NDArray::from_shape_vec([2, 2], vec![0_i32, 1, 2, 3]).unwrap();
        let rhs = NDArray::from_shape_vec([2, 2], vec![4_i32, 5, 6, 7]).unwrap();
        let stacked = NDArray::stack(&[lhs.view(), rhs.view()], 1).unwrap();

        assert_eq!(stacked.shape(), &[2, 2, 2]);
        assert_eq!(stacked.strides(), &[4, 2, 1]);
        assert_eq!(stacked.data(), &[0, 1, 4, 5, 2, 3, 6, 7]);
    }

    #[test]
    fn stack_supports_negative_axes_and_strided_views() {
        let lhs = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let rhs = NDArray::from_shape_vec([2, 3], vec![6_i32, 7, 8, 9, 10, 11]).unwrap();
        let stacked =
            NDArray::stack(&[lhs.view().transpose(), rhs.view().transpose()], -1).unwrap();

        assert_eq!(stacked.shape(), &[3, 2, 2]);
        assert_eq!(stacked.strides(), &[4, 2, 1]);
        assert_eq!(stacked.data(), &[0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]);
    }

    #[test]
    fn stack_validates_non_empty_inputs_axes_and_shapes() {
        let lhs = NDArray::from_shape_vec([2, 2], vec![0_i32, 1, 2, 3]).unwrap();
        let rhs = NDArray::from_shape_vec([2, 3], vec![4_i32, 5, 6, 7, 8, 9]).unwrap();
        let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();

        assert_eq!(
            NDArray::<i32>::stack(&[], 0).unwrap_err(),
            AtlasNdError::InvalidArgument { op: "stack", reason: "at least one array is required" }
        );
        assert_eq!(
            NDArray::stack(&[lhs.view(), rhs.view()], 0).unwrap_err(),
            AtlasNdError::InvalidArgument { op: "stack", reason: "all input shapes must match" }
        );
        assert_eq!(
            NDArray::stack(&[scalar.view()], 1).unwrap_err(),
            AtlasNdError::InvalidAxis { axis: 1, ndim: 0 }
        );
    }

    #[test]
    fn stack_preserves_zero_length_shapes() {
        let lhs = NDArray::<i32>::from_shape_vec([2, 0, 3], Vec::new()).unwrap();
        let rhs = NDArray::<i32>::from_shape_vec([2, 0, 3], Vec::new()).unwrap();
        let stacked = NDArray::stack(&[lhs.view(), rhs.view()], 1).unwrap();

        assert_eq!(stacked.shape(), &[2, 2, 0, 3]);
        assert_eq!(stacked.strides(), &[0, 0, 3, 1]);
        assert!(stacked.data().is_empty());
    }
}
