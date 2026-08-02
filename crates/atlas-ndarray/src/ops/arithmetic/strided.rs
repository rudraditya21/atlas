use crate::{
    NDArray, Numeric,
    internal::offset_pair_iter,
    layout::{broadcast::BroadcastMetadata, element_count},
};

use super::from_owned_parts;

pub(super) fn elementwise_binary_strided<T, F>(
    lhs: &NDArray<T>,
    rhs: &NDArray<T>,
    metadata: BroadcastMetadata,
    op: F,
) -> NDArray<T>
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
{
    let output_len = element_count(&metadata.shape);
    let mut data = vec![T::zero(); output_len];

    for (slot, (lhs_offset, rhs_offset)) in data.iter_mut().zip(offset_pair_iter(
        0,
        0,
        &metadata.shape,
        &metadata.lhs_strides,
        &metadata.rhs_strides,
    )) {
        *slot = op(lhs.data()[lhs_offset], rhs.data()[rhs_offset]);
    }

    from_owned_parts(metadata.shape, data)
}
