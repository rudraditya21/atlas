use super::{
    broadcast::elementwise_binary_broadcast,
    contiguous::elementwise_binary_contiguous,
    scalar::{elementwise_scalar_lhs, elementwise_scalar_rhs},
    strided::elementwise_binary_strided,
};
use crate::{
    AtlasNdResult, NDArray, Numeric,
    internal::layout::{PairLayoutKind, pair_layout_kind},
    layout::broadcast::{BroadcastMetadata, broadcast_pair},
};

pub(super) enum BinaryOperand<'a, T: Numeric> {
    Array(&'a NDArray<T>),
    Scalar(T),
}

struct BinaryArrayDispatch<'lhs, 'rhs, T: Numeric> {
    lhs: &'lhs NDArray<T>,
    rhs: &'rhs NDArray<T>,
    layout_kind: PairLayoutKind,
    metadata: BroadcastMetadata,
}

enum NormalizedBinaryOperands<'lhs, 'rhs, T: Numeric> {
    ScalarRhs(T),
    ScalarLhs { scalar: T, array: &'rhs NDArray<T> },
    Arrays(BinaryArrayDispatch<'lhs, 'rhs, T>),
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
    let operands = normalize_binary_operands(lhs, rhs)?;

    match operands {
        NormalizedBinaryOperands::ScalarRhs(scalar) => Ok(scalar_rhs_op(lhs, scalar)),
        NormalizedBinaryOperands::ScalarLhs { scalar, array } => Ok(scalar_lhs_op(scalar, array)),
        NormalizedBinaryOperands::Arrays(dispatch) => match dispatch.layout_kind {
            PairLayoutKind::Contiguous => Ok(contiguous_op(dispatch.lhs, dispatch.rhs)),
            PairLayoutKind::Broadcast => {
                Ok(elementwise_binary_broadcast(dispatch.lhs, dispatch.rhs, dispatch.metadata, op))
            }
            PairLayoutKind::Strided => {
                Ok(elementwise_binary_strided(dispatch.lhs, dispatch.rhs, dispatch.metadata, op))
            }
        },
    }
}

fn normalize_binary_operands<'lhs, 'rhs, T: Numeric>(
    lhs: &'lhs NDArray<T>,
    rhs: BinaryOperand<'rhs, T>,
) -> AtlasNdResult<NormalizedBinaryOperands<'lhs, 'rhs, T>> {
    match rhs {
        BinaryOperand::Scalar(scalar) => Ok(NormalizedBinaryOperands::ScalarRhs(scalar)),
        BinaryOperand::Array(rhs_array) => {
            if rhs_array.shape().is_empty() {
                return Ok(NormalizedBinaryOperands::ScalarRhs(rhs_array.data()[0]));
            }

            if lhs.shape().is_empty() {
                return Ok(NormalizedBinaryOperands::ScalarLhs {
                    scalar: lhs.data()[0],
                    array: rhs_array,
                });
            }

            normalize_binary_arrays(lhs, rhs_array)
        }
    }
}

fn normalize_binary_arrays<'lhs, 'rhs, T: Numeric>(
    lhs: &'lhs NDArray<T>,
    rhs: &'rhs NDArray<T>,
) -> AtlasNdResult<NormalizedBinaryOperands<'lhs, 'rhs, T>> {
    let metadata = broadcast_pair(lhs.shape(), lhs.strides(), rhs.shape(), rhs.strides())?;
    let layout_kind =
        pair_layout_kind(&metadata.shape, &metadata.lhs_strides, &metadata.rhs_strides);

    Ok(NormalizedBinaryOperands::Arrays(BinaryArrayDispatch { lhs, rhs, layout_kind, metadata }))
}
