use crate::{
    ArrayElement, AtlasNdError, AtlasNdResult, NDArray, OperandMetadata,
    internal::logical_span_iter,
};

impl<T: ArrayElement> NDArray<T> {
    /// Replaces values whose corresponding mask values are true.
    pub fn masked_fill<M>(&mut self, mask: &M, value: T) -> AtlasNdResult<()>
    where
        M: OperandMetadata<bool> + ?Sized,
    {
        if self.shape() != mask.shape() {
            return Err(AtlasNdError::MaskShapeMismatch {
                op: "masked_fill",
                array: self.shape().to_vec(),
                mask: mask.shape().to_vec(),
            });
        }

        let mut remaining = self.data.as_mut_slice();
        for selections in
            logical_span_iter(mask.data(), mask.offset(), mask.shape(), mask.strides())
        {
            let (values, next) = remaining.split_at_mut(selections.len());
            for (slot, &selected) in values.iter_mut().zip(selections) {
                if selected {
                    *slot = value;
                }
            }
            remaining = next;
        }
        debug_assert!(remaining.is_empty());

        Ok(())
    }
}
