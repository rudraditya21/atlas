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
mod view;

pub use core::array::NDArray;
pub use core::asarray::AsArray;
pub use core::axis::AxisIndex;
pub use core::dtype::{
    ArithmeticPromote, CastMode, CastPolicy, DType, ReductionOp, RuntimeDType, RuntimeScalar,
    ScalarValue, infer_scalar_dtype,
};
pub use core::error::{AtlasNdError, AtlasNdResult};
pub use core::operand::OperandMetadata;
pub use core::traits::{ArrayElement, Numeric, ShapeArg};
pub use internal::shape::{
    checked_compute_strides, checked_element_count, compute_strides, element_count,
};
pub use layout::broadcast::{
    BroadcastMetadata, broadcast_pair, broadcast_shape, broadcast_strides,
    contiguous_broadcast_metadata,
};
pub use ops::arithmetic::{
    AddOperand, DivOperand, ElementwiseArithmetic, ElementwiseMinMax, MaxOperand, MinOperand,
    MulOperand, RemOperand, SubOperand,
};
pub use ops::bitwise::{BitwiseElement, BitwiseOperand};
pub use ops::comparison::{EqOperand, GeOperand, GtOperand, LeOperand, LtOperand, NeOperand};
pub use ops::logical::LogicalOperand;
pub use ops::unary::{FloatClassify, UnaryAbs, UnaryNeg, UnaryRound, UnarySign};
pub use ops::where_::{IntoWhereOperand, WhereOperand};
pub use view::iter::ArrayViewIter;
pub use view::slicing::SliceRange;
pub use view::view::ArrayView;
