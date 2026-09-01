use crate::{
    ArrayElement, AtlasNdError, AtlasNdResult, NDArray, OperandMetadata, checked_compute_strides,
    checked_element_count, internal::value_iter, view::ArrayView,
};

impl<T: ArrayElement> NDArray<T> {
    /// Returns an owned array padded with `value` before and after each axis.
    pub fn pad<P: AsRef<[(usize, usize)]>>(&self, widths: P, value: T) -> AtlasNdResult<Self> {
        pad_operand(self, widths.as_ref(), value)
    }
}

impl<'a, T: ArrayElement> ArrayView<'a, T> {
    /// Returns an owned array padded with `value` before and after each axis.
    pub fn pad<P: AsRef<[(usize, usize)]>>(
        &self,
        widths: P,
        value: T,
    ) -> AtlasNdResult<NDArray<T>> {
        pad_operand(self, widths.as_ref(), value)
    }
}

fn pad_operand<T, O>(operand: &O, widths: &[(usize, usize)], value: T) -> AtlasNdResult<NDArray<T>>
where
    T: ArrayElement,
    O: OperandMetadata<T> + ?Sized,
{
    if widths.len() != operand.ndim() {
        return Err(AtlasNdError::DimensionMismatch {
            expected: operand.ndim(),
            actual: widths.len(),
        });
    }

    let shape: Vec<_> = operand
        .shape()
        .iter()
        .zip(widths)
        .map(|(&dimension, &(before, after))| {
            dimension
                .checked_add(before)
                .and_then(|dimension| dimension.checked_add(after))
                .ok_or_else(|| AtlasNdError::ShapeOverflow {
                    op: "pad",
                    shape: operand.shape().to_vec(),
                })
        })
        .collect::<AtlasNdResult<_>>()?;
    let output_len = checked_element_count(&shape)?;
    let input_strides = checked_compute_strides(operand.shape())?;
    let output_strides = checked_compute_strides(&shape)?;
    let mut data = vec![value; output_len];

    for (input_index, element) in
        value_iter(operand.data(), operand.offset(), operand.shape(), operand.strides())
            .copied()
            .enumerate()
    {
        let mut output_index = 0;
        for axis in 0..operand.ndim() {
            let coordinate = input_index / input_strides[axis] % operand.shape()[axis];
            output_index += (coordinate + widths[axis].0) * output_strides[axis];
        }
        data[output_index] = element;
    }

    NDArray::from_shape_vec(shape, data)
}
