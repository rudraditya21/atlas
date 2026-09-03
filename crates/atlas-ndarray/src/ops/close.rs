use num_traits::Float;

use crate::{
    AtlasNdError, AtlasNdResult, Numeric, OperandMetadata, internal::broadcast_offset_pair_iter,
    layout::broadcast::broadcast_pair,
};

/// Returns whether all broadcasted element pairs satisfy `|lhs - rhs| <= atol + rtol * |rhs|`.
///
/// Matching infinities compare equal. NaNs compare unequal unless `equal_nan` is `true` and both
/// values in a pair are NaN. Tolerances must be finite and non-negative.
pub fn allclose<T, L, R>(lhs: &L, rhs: &R, rtol: T, atol: T, equal_nan: bool) -> AtlasNdResult<bool>
where
    T: Numeric + Float,
    L: OperandMetadata<T> + ?Sized,
    R: OperandMetadata<T> + ?Sized,
{
    if !rtol.is_finite() || rtol < T::zero() || !atol.is_finite() || atol < T::zero() {
        return Err(AtlasNdError::InvalidArgument {
            op: "allclose",
            reason: "tolerances must be finite and non-negative",
        });
    }

    let metadata = broadcast_pair(lhs.shape(), lhs.strides(), rhs.shape(), rhs.strides())?;

    Ok(broadcast_offset_pair_iter(
        lhs.offset(),
        rhs.offset(),
        &metadata.shape,
        &metadata.lhs_strides,
        &metadata.rhs_strides,
    )
    .all(|(lhs_offset, rhs_offset)| {
        let left = lhs.data()[lhs_offset];
        let right = rhs.data()[rhs_offset];

        left == right
            || (equal_nan && left.is_nan() && right.is_nan())
            || (!left.is_nan()
                && !right.is_nan()
                && (left - right).abs() <= atol + rtol * right.abs())
    }))
}
