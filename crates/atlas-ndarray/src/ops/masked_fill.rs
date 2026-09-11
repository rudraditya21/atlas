use crate::{
    ArrayElement, AtlasNdError, AtlasNdResult, NDArray, OperandMetadata, internal::offset_iter,
};

impl<T: ArrayElement> NDArray<T> {
    /// Replaces values whose corresponding mask values are true.
    pub fn masked_fill<M>(&mut self, mask: &M, value: T) -> AtlasNdResult<()>
    where
        M: OperandMetadata<bool> + ?Sized,
    {
        if self.shape() != mask.shape() {
            return Err(AtlasNdError::InvalidArgument {
                op: "masked_fill",
                reason: "mask shape must match array shape",
            });
        }

        for (slot, mask_offset) in
            self.data.iter_mut().zip(offset_iter(mask.offset(), mask.shape(), mask.strides()))
        {
            if mask.data()[mask_offset] {
                *slot = value;
            }
        }

        Ok(())
    }
}
