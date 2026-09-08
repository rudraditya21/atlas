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
