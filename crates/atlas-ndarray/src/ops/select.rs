use crate::{
    ArrayElement, AtlasNdError, AtlasNdResult, NDArray, OperandMetadata,
    internal::{logical_span_iter, value_iter},
    view::ArrayView,
};

impl<T: ArrayElement> NDArray<T> {
    /// Returns logical values whose corresponding mask values are true as a contiguous rank-1 array.
    pub fn select<M>(&self, mask: &M) -> AtlasNdResult<NDArray<T>>
    where
        M: OperandMetadata<bool> + ?Sized,
    {
        select_operand(self, mask)
    }
}

impl<T: ArrayElement> ArrayView<'_, T> {
    /// Returns logical values whose corresponding mask values are true as a contiguous rank-1 array.
    pub fn select<M>(&self, mask: &M) -> AtlasNdResult<NDArray<T>>
    where
        M: OperandMetadata<bool> + ?Sized,
    {
        select_operand(self, mask)
    }
}

fn select_operand<T, O, M>(operand: &O, mask: &M) -> AtlasNdResult<NDArray<T>>
where
    T: ArrayElement,
    O: OperandMetadata<T> + ?Sized,
    M: OperandMetadata<bool> + ?Sized,
{
    if operand.shape() != mask.shape() {
        return Err(AtlasNdError::InvalidArgument {
            op: "select",
            reason: "mask shape must match array shape",
        });
    }

    let mut data = Vec::with_capacity(count_selected(mask));

    if operand.strides() == mask.strides() {
        for (values, selections) in
            logical_span_iter(operand.data(), operand.offset(), operand.shape(), operand.strides())
                .zip(logical_span_iter(mask.data(), mask.offset(), mask.shape(), mask.strides()))
        {
            debug_assert_eq!(values.len(), selections.len());
            data.extend(
                values
                    .iter()
                    .copied()
                    .zip(selections.iter().copied())
                    .filter_map(|(value, selected)| selected.then_some(value)),
            );
        }
    } else {
        data.extend(
            value_iter(operand.data(), operand.offset(), operand.shape(), operand.strides())
                .copied()
                .zip(value_iter(mask.data(), mask.offset(), mask.shape(), mask.strides()).copied())
                .filter_map(|(value, selected)| selected.then_some(value)),
        );
    }

    Ok(NDArray::from_shape_vec([data.len()], data)?)
}

fn count_selected<M: OperandMetadata<bool> + ?Sized>(mask: &M) -> usize {
    logical_span_iter(mask.data(), mask.offset(), mask.shape(), mask.strides())
        .map(|span| span.iter().filter(|&&selected| selected).count())
        .sum()
}
