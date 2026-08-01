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
pub use core::axis::AxisIndex;
pub use core::error::{AtlasNdError, AtlasNdResult};
pub use core::traits::Numeric;
pub use layout::{
    BroadcastMetadata, broadcast_pair, broadcast_shape, broadcast_strides, compute_strides,
    contiguous_broadcast_metadata, element_count,
};
pub use ops::{AddOperand, DivOperand, MulOperand, SubOperand};
pub use view::{ArrayView, ArrayViewIter};
