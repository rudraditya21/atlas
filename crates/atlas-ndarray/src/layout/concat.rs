use crate::{
    ArrayElement, AtlasNdError, AtlasNdResult, AxisIndex, NDArray, core::axis::normalize_axis,
    internal::value_iter, view::ArrayView,
};

impl<T: ArrayElement> NDArray<T> {
    pub fn concatenate<A: AxisIndex>(arrays: &[ArrayView<'_, T>], axis: A) -> AtlasNdResult<Self> {
        concatenate_impl(arrays, axis)
    }
}

fn concatenate_impl<T: ArrayElement, A: AxisIndex>(
    arrays: &[ArrayView<'_, T>],
    axis: A,
) -> AtlasNdResult<NDArray<T>> {
    let Some(first) = arrays.first() else {
        return Err(AtlasNdError::InvalidArgument {
            op: "concatenate",
            reason: "at least one array is required",
        });
    };

    let ndim = first.ndim();
    let axis = normalize_axis(axis, ndim)?;
    let mut output_shape = first.shape().to_vec();

    for array in &arrays[1..] {
        if array.ndim() != ndim {
            return Err(AtlasNdError::DimensionMismatch { expected: ndim, actual: array.ndim() });
        }

        for (current_axis, (&expected, &actual)) in
            first.shape().iter().zip(array.shape().iter()).enumerate()
        {
            if current_axis != axis && expected != actual {
                return Err(AtlasNdError::InvalidArgument {
                    op: "concatenate",
                    reason: "non-concatenation dimensions must match",
                });
            }
        }

        output_shape[axis] =
            output_shape[axis].checked_add(array.shape()[axis]).ok_or_else(|| {
                AtlasNdError::ShapeOverflow { op: "concatenate", shape: output_shape.clone() }
            })?;
    }

    let total_len = crate::checked_element_count(&output_shape)?;
    let mut data = Vec::with_capacity(total_len);
    let outer_len = crate::element_count(&output_shape[..axis]);

    for outer_index in 0..outer_len {
        for array in arrays {
            let block_offset = prefix_offset(
                array.offset(),
                &array.shape()[..axis],
                &array.strides()[..axis],
                outer_index,
            );
            data.extend(
                value_iter(
                    array.data(),
                    block_offset,
                    &array.shape()[axis..],
                    &array.strides()[axis..],
                )
                .copied(),
            );
        }
    }

    NDArray::from_row_major_parts(output_shape, data)
}

fn prefix_offset(
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    mut linear_index: usize,
) -> usize {
    let mut offset = base_offset;

    for axis in (0..shape.len()).rev() {
        let dim = shape[axis];
        let coordinate = if dim == 0 { 0 } else { linear_index % dim };
        linear_index = linear_index.checked_div(dim).unwrap_or(0);
        offset += coordinate * strides[axis];
    }

    offset
}

#[cfg(test)]
mod tests {
    use crate::{AtlasNdError, NDArray};

    #[test]
    fn concatenate_joins_contiguous_arrays_along_first_axis() {
        let lhs = NDArray::from_shape_vec([2, 2], vec![0_i32, 1, 2, 3]).unwrap();
        let rhs = NDArray::from_shape_vec([1, 2], vec![4_i32, 5]).unwrap();
        let joined = NDArray::concatenate(&[lhs.view(), rhs.view()], 0).unwrap();

        assert_eq!(joined.shape(), &[3, 2]);
        assert_eq!(joined.strides(), &[2, 1]);
        assert_eq!(joined.data(), &[0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn concatenate_interleaves_row_major_blocks_for_nonzero_axes() {
        let lhs = NDArray::from_shape_vec([2, 2], vec![0_i32, 1, 2, 3]).unwrap();
        let rhs = NDArray::from_shape_vec([2, 1], vec![4_i32, 5]).unwrap();
        let joined = NDArray::concatenate(&[lhs.view(), rhs.view()], -1).unwrap();

        assert_eq!(joined.shape(), &[2, 3]);
        assert_eq!(joined.strides(), &[3, 1]);
        assert_eq!(joined.data(), &[0, 1, 4, 2, 3, 5]);
    }

    #[test]
    fn concatenate_materializes_strided_views_in_logical_order() {
        let lhs = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let rhs = NDArray::from_shape_vec([2, 3], vec![6_i32, 7, 8, 9, 10, 11]).unwrap();
        let joined =
            NDArray::concatenate(&[lhs.view().transpose(), rhs.view().transpose()], 1).unwrap();

        assert_eq!(joined.shape(), &[3, 4]);
        assert_eq!(joined.strides(), &[4, 1]);
        assert_eq!(joined.data(), &[0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]);
    }

    #[test]
    fn concatenate_validates_empty_inputs_axes_and_shapes() {
        let lhs = NDArray::from_shape_vec([2, 2], vec![0_i32, 1, 2, 3]).unwrap();
        let rhs = NDArray::from_shape_vec([3, 2], vec![4_i32, 5, 6, 7, 8, 9]).unwrap();
        let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();

        assert_eq!(
            NDArray::<i32>::concatenate(&[], 0).unwrap_err(),
            AtlasNdError::InvalidArgument {
                op: "concatenate",
                reason: "at least one array is required",
            }
        );
        assert_eq!(
            NDArray::concatenate(&[scalar.view()], 0).unwrap_err(),
            AtlasNdError::InvalidAxis { axis: 0, ndim: 0 }
        );
        assert_eq!(
            NDArray::concatenate(&[lhs.view(), rhs.view()], 1).unwrap_err(),
            AtlasNdError::InvalidArgument {
                op: "concatenate",
                reason: "non-concatenation dimensions must match",
            }
        );
    }

    #[test]
    fn concatenate_preserves_zero_length_outputs_when_non_joined_dimensions_are_empty() {
        let lhs = NDArray::<i32>::from_shape_vec([2, 0, 3], Vec::new()).unwrap();
        let rhs = NDArray::<i32>::from_shape_vec([2, 0, 5], Vec::new()).unwrap();
        let joined = NDArray::concatenate(&[lhs.view(), rhs.view()], 2).unwrap();

        assert_eq!(joined.shape(), &[2, 0, 8]);
        assert_eq!(joined.strides(), &[0, 8, 1]);
        assert!(joined.data().is_empty());
    }
}
