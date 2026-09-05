use crate::{
    ArrayElement, ArrayView, AtlasNdResult, AxisIndex, NDArray, OperandMetadata,
    checked_element_count,
    core::axis::{normalize_axis, normalize_scalar_index},
};

fn take_operand<T: ArrayElement, O: OperandMetadata<T> + ?Sized, I: AxisIndex, A: AxisIndex>(
    operand: &O,
    indices: &[I],
    axis: A,
) -> AtlasNdResult<NDArray<T>> {
    let axis = normalize_axis(axis, operand.ndim())?;
    let selected = indices
        .iter()
        .map(|index| normalize_scalar_index(*index, axis, operand.shape()[axis]))
        .collect::<AtlasNdResult<Vec<_>>>()?;
    let mut shape = operand.shape().to_vec();
    shape[axis] = selected.len();
    let output_len = checked_element_count(&shape)?;
    let mut data = Vec::with_capacity(output_len);

    for output_index in 0..output_len {
        let mut remainder = output_index;
        let mut offset = operand.offset();
        for dimension in (0..shape.len()).rev() {
            let coordinate = remainder % shape[dimension];
            remainder /= shape[dimension];
            let input_coordinate =
                if dimension == axis { selected[coordinate] } else { coordinate };
            offset += input_coordinate * operand.strides()[dimension];
        }
        data.push(operand.data()[offset]);
    }
    NDArray::from_row_major_parts(shape, data)
}

macro_rules! impl_take {
    ($operand:ty) => {
        impl<T: ArrayElement> $operand {
            /// Gathers elements at `indices` along `axis` into an owned row-major array.
            ///
            /// Negative indices address from the end of the selected axis. Every index is
            /// validated before gathering begins.
            pub fn take<I: AxisIndex, A: AxisIndex>(
                &self,
                indices: &[I],
                axis: A,
            ) -> AtlasNdResult<NDArray<T>> {
                take_operand(self, indices, axis)
            }
        }
    };
}

impl_take!(NDArray<T>);
impl_take!(ArrayView<'_, T>);

#[cfg(test)]
mod tests {
    use crate::{AtlasNdError, NDArray};

    #[test]
    fn take_gathers_selected_axis_in_requested_order() {
        let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let taken = array.take(&[2, 0], 1_i32).unwrap();

        assert_eq!(taken.shape(), &[2, 2]);
        assert_eq!(taken.data(), &[2, 0, 5, 3]);
    }

    #[test]
    fn take_supports_negative_indices_and_views() {
        let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let taken = array.view().transpose().take(&[-1, 0], 0_i32).unwrap();

        assert_eq!(taken.shape(), &[2, 2]);
        assert_eq!(taken.data(), &[2, 5, 0, 3]);
    }

    #[test]
    fn take_validates_axis_and_indices() {
        let array = NDArray::from_shape_vec([2, 3], vec![0_i32; 6]).unwrap();

        assert_eq!(
            array.take(&[3], 1_i32).unwrap_err(),
            AtlasNdError::IndexOutOfBounds { axis: 1, index: 3, dim: 3 }
        );
        assert_eq!(
            array.take(&[0], 2_i32).unwrap_err(),
            AtlasNdError::InvalidAxis { axis: 2, ndim: 2 }
        );
    }
}
