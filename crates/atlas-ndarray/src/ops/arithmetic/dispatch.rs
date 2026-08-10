use crate::{
    AtlasNdResult, NDArray, Numeric,
    internal::layout::{PairLayoutKind, pair_layout_kind},
    layout::broadcast::{BroadcastMetadata, broadcast_pair},
};

use super::{
    broadcast::elementwise_binary_broadcast,
    contiguous::elementwise_binary_contiguous,
    scalar::{elementwise_scalar_lhs, elementwise_scalar_rhs},
    strided::elementwise_binary_strided,
};

pub(super) enum BinaryOperand<'a, T: Numeric> {
    Array(&'a NDArray<T>),
    Scalar(T),
}

struct BinaryArrayDispatch<'a, T: Numeric> {
    rhs: &'a NDArray<T>,
    layout_kind: PairLayoutKind,
    metadata: BroadcastMetadata,
}

enum BinaryDispatch<'a, T: Numeric> {
    ScalarRhs(T),
    ScalarLhs { scalar: T, array: &'a NDArray<T> },
    Arrays(BinaryArrayDispatch<'a, T>),
}

pub(super) fn dispatch_elementwise_binary<T, F>(
    lhs: &NDArray<T>,
    rhs: BinaryOperand<'_, T>,
    op: F,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
{
    dispatch_elementwise_binary_with(
        lhs,
        rhs,
        |array, scalar| elementwise_scalar_rhs(array, scalar, op),
        |scalar, array| elementwise_scalar_lhs(scalar, array, op),
        |lhs, rhs| elementwise_binary_contiguous(lhs, rhs, op),
        op,
    )
}

pub(super) fn dispatch_elementwise_binary_with<T, F, SR, SL, C>(
    lhs: &NDArray<T>,
    rhs: BinaryOperand<'_, T>,
    scalar_rhs_op: SR,
    scalar_lhs_op: SL,
    contiguous_op: C,
    op: F,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
    SR: Fn(&NDArray<T>, T) -> NDArray<T>,
    SL: Fn(T, &NDArray<T>) -> NDArray<T>,
    C: Fn(&NDArray<T>, &NDArray<T>) -> NDArray<T>,
{
    let dispatch = classify_binary_dispatch(lhs, rhs)?;

    match dispatch {
        BinaryDispatch::ScalarRhs(scalar) => Ok(scalar_rhs_op(lhs, scalar)),
        BinaryDispatch::ScalarLhs { scalar, array } => Ok(scalar_lhs_op(scalar, array)),
        BinaryDispatch::Arrays(dispatch) => match dispatch.layout_kind {
            PairLayoutKind::Contiguous => Ok(contiguous_op(lhs, dispatch.rhs)),
            PairLayoutKind::Broadcast => {
                Ok(elementwise_binary_broadcast(lhs, dispatch.rhs, dispatch.metadata, op))
            }
            PairLayoutKind::Strided => {
                Ok(elementwise_binary_strided(lhs, dispatch.rhs, dispatch.metadata, op))
            }
        },
    }
}

fn classify_binary_dispatch<'a, T: Numeric>(
    lhs: &NDArray<T>,
    rhs: BinaryOperand<'a, T>,
) -> AtlasNdResult<BinaryDispatch<'a, T>> {
    match rhs {
        BinaryOperand::Scalar(scalar) => Ok(BinaryDispatch::ScalarRhs(scalar)),
        BinaryOperand::Array(rhs_array) => {
            if rhs_array.shape().is_empty() {
                return Ok(BinaryDispatch::ScalarRhs(rhs_array.data()[0]));
            }

            if lhs.shape().is_empty() {
                return Ok(BinaryDispatch::ScalarLhs { scalar: lhs.data()[0], array: rhs_array });
            }

            classify_binary_arrays(lhs, rhs_array)
        }
    }
}

fn classify_binary_arrays<'a, T: Numeric>(
    lhs: &NDArray<T>,
    rhs: &'a NDArray<T>,
) -> AtlasNdResult<BinaryDispatch<'a, T>> {
    let metadata = broadcast_pair(lhs.shape(), lhs.strides(), rhs.shape(), rhs.strides())?;
    let layout_kind =
        pair_layout_kind(&metadata.shape, &metadata.lhs_strides, &metadata.rhs_strides);

    Ok(BinaryDispatch::Arrays(BinaryArrayDispatch { rhs, layout_kind, metadata }))
}
