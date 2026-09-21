//! `atlas-ndarray` is the dense tensor core for Atlas.
//!
//! This crate currently targets CPU-backed dense arrays and views with:
//! - row-major contiguous owned arrays
//! - metadata-only views, slicing, transpose, and reshape validation
//! - broadcast-aware elementwise operations
//! - whole-array reductions
//!
//! Public invariants:
//! - `shape.len() == strides.len()`
//! - owned arrays created by constructors are row-major contiguous
//! - `data.len() == product(shape)` for owned arrays
//! - views preserve logical element mapping through offset + shape + strides
//! - reshape is allowed only when the view is contiguous and element count is unchanged
//! - broadcast compatibility follows trailing-dimension alignment with singleton expansion

mod constructors;
mod core;
pub(crate) mod internal;
mod layout;
mod ops;
#[doc(hidden)]
pub mod simd_support;
mod view;

pub use core::{
    array::NDArray,
    asarray::AsArray,
    axis::AxisIndex,
    dtype::{
        ArithmeticPromote, CastMode, CastPolicy, DType, ReductionOp, RuntimeDType, RuntimeScalar,
        ScalarValue, infer_scalar_dtype,
    },
    error::{AtlasNdError, AtlasNdResult},
    operand::OperandMetadata,
    traits::{ArrayElement, Numeric, ShapeArg},
};

pub use internal::shape::{
    checked_compute_strides, checked_element_count, compute_strides, element_count,
};
pub use layout::broadcast::{
    BroadcastMetadata, broadcast_pair, broadcast_shape, broadcast_strides,
    contiguous_broadcast_metadata,
};
pub use ops::{
    arithmetic::{
        AddOperand, DivOperand, ElementwiseArithmetic, ElementwiseDivision, ElementwiseMinMax,
        MaxOperand, MinOperand, MulOperand, RemOperand, SubOperand,
    },
    bitwise::{BitwiseElement, BitwiseOperand},
    close::allclose,
    comparison::{EqOperand, GeOperand, GtOperand, LeOperand, LtOperand, NeOperand},
    indexing::Truthy,
    logical::LogicalOperand,
    unary::{FloatClassify, UnaryAbs, UnaryNeg, UnaryRound, UnarySign},
    where_::{IntoWhereOperand, WhereOperand},
};
pub use view::{iter::ArrayViewIter, slicing::SliceRange, view::ArrayView};

/// Visits maximal contiguous storage spans in an operand's logical row-major order.
///
/// This is primarily useful to downstream Atlas crates that need to process logical views
/// without resolving a multidimensional coordinate for every value.
#[doc(hidden)]
pub fn try_for_each_logical_span<T, O, E, F>(operand: &O, mut f: F) -> Result<(), E>
where
    T: ArrayElement,
    O: OperandMetadata<T> + ?Sized,
    F: FnMut(&[T]) -> Result<(), E>,
{
    for span in internal::traversal::logical_span_iter(
        operand.data(),
        operand.offset(),
        operand.shape(),
        operand.strides(),
    ) {
        f(span)?;
    }

    Ok(())
}

/// Visits paired contiguous spans from operands with the same logical element count.
#[doc(hidden)]
pub fn try_for_each_logical_span_pair<T, L, R, E, F>(lhs: &L, rhs: &R, mut f: F) -> Result<(), E>
where
    T: ArrayElement,
    L: OperandMetadata<T> + ?Sized,
    R: OperandMetadata<T> + ?Sized,
    E: From<AtlasNdError>,
    F: FnMut(&[T], &[T]) -> Result<(), E>,
{
    let lhs_len = checked_element_count(lhs.shape()).map_err(E::from)?;
    let rhs_len = checked_element_count(rhs.shape()).map_err(E::from)?;
    if lhs_len != rhs_len {
        return Err(E::from(AtlasNdError::ShapeMismatch { expected: lhs_len, actual: rhs_len }));
    }

    let mut lhs_spans = internal::traversal::logical_span_iter(
        lhs.data(),
        lhs.offset(),
        lhs.shape(),
        lhs.strides(),
    );
    let mut rhs_spans = internal::traversal::logical_span_iter(
        rhs.data(),
        rhs.offset(),
        rhs.shape(),
        rhs.strides(),
    );
    let mut lhs_span = lhs_spans.next();
    let mut rhs_span = rhs_spans.next();
    let mut lhs_index = 0;
    let mut rhs_index = 0;

    while let (Some(lhs_span_values), Some(rhs_span_values)) = (lhs_span, rhs_span) {
        let span_len = (lhs_span_values.len() - lhs_index).min(rhs_span_values.len() - rhs_index);
        f(
            &lhs_span_values[lhs_index..lhs_index + span_len],
            &rhs_span_values[rhs_index..rhs_index + span_len],
        )?;

        lhs_index += span_len;
        rhs_index += span_len;
        if lhs_index == lhs_span_values.len() {
            lhs_span = lhs_spans.next();
            lhs_index = 0;
        }
        if rhs_index == rhs_span_values.len() {
            rhs_span = rhs_spans.next();
            rhs_index = 0;
        }
    }

    debug_assert!(lhs_span.is_none() && rhs_span.is_none());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{AtlasNdError, NDArray, try_for_each_logical_span_pair};

    #[test]
    fn paired_span_traversal_rejects_mismatched_logical_lengths() {
        let lhs = NDArray::from_shape_vec([2], vec![1_i32, 2]).unwrap();
        let rhs = NDArray::from_shape_vec([3], vec![3_i32, 4, 5]).unwrap();
        let mut called = false;

        let error = try_for_each_logical_span_pair(&lhs, &rhs, |_, _| {
            called = true;
            Ok::<_, AtlasNdError>(())
        })
        .unwrap_err();

        assert!(!called);
        assert_eq!(error, AtlasNdError::ShapeMismatch { expected: 2, actual: 3 });
    }
}
