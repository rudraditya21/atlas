use crate::{
    AtlasNdResult, NDArray, Numeric,
    internal::layout::{PairLayoutKind, pair_layout_kind},
    layout::broadcast::{BroadcastMetadata, broadcast_pair},
};

use super::{
    broadcast::elementwise_binary_broadcast, contiguous::elementwise_binary_contiguous,
    strided::elementwise_binary_strided,
};

struct BinaryDispatchMetadata {
    layout_kind: PairLayoutKind,
    metadata: BroadcastMetadata,
}

pub(super) fn dispatch_elementwise_binary<T, F>(
    lhs: &NDArray<T>,
    rhs: &NDArray<T>,
    op: F,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
{
    let dispatch = classify_binary_dispatch(lhs, rhs)?;

    match dispatch.layout_kind {
        PairLayoutKind::Contiguous => Ok(elementwise_binary_contiguous(lhs, rhs, op)),
        PairLayoutKind::Broadcast => {
            Ok(elementwise_binary_broadcast(lhs, rhs, dispatch.metadata, op))
        }
        PairLayoutKind::Strided => Ok(elementwise_binary_strided(lhs, rhs, dispatch.metadata, op)),
    }
}

pub(super) fn dispatch_elementwise_binary_with_contiguous<T, F, C>(
    lhs: &NDArray<T>,
    rhs: &NDArray<T>,
    contiguous_op: C,
    op: F,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
    C: Fn(&NDArray<T>, &NDArray<T>) -> NDArray<T>,
{
    let dispatch = classify_binary_dispatch(lhs, rhs)?;

    match dispatch.layout_kind {
        PairLayoutKind::Contiguous => Ok(contiguous_op(lhs, rhs)),
        PairLayoutKind::Broadcast => {
            Ok(elementwise_binary_broadcast(lhs, rhs, dispatch.metadata, op))
        }
        PairLayoutKind::Strided => Ok(elementwise_binary_strided(lhs, rhs, dispatch.metadata, op)),
    }
}

fn classify_binary_dispatch<T: Numeric>(
    lhs: &NDArray<T>,
    rhs: &NDArray<T>,
) -> AtlasNdResult<BinaryDispatchMetadata> {
    let metadata = broadcast_pair(lhs.shape(), lhs.strides(), rhs.shape(), rhs.strides())?;
    let layout_kind =
        pair_layout_kind(&metadata.shape, &metadata.lhs_strides, &metadata.rhs_strides);

    Ok(BinaryDispatchMetadata { layout_kind, metadata })
}
