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

mod arithmetic;
mod array;
mod axis;
mod broadcast;
mod constructors;
mod error;
mod indexing;
mod iter;
mod reduction;
mod reshape;
mod slicing;
mod stride;
mod traits;
mod transpose;
pub(crate) mod traversal;
mod view;

pub use arithmetic::{AddOperand, DivOperand, MulOperand, SubOperand};
pub use array::NDArray;
pub use axis::AxisIndex;
pub use broadcast::{
    BroadcastMetadata, broadcast_pair, broadcast_shape, broadcast_strides,
    contiguous_broadcast_metadata,
};
pub use error::{AtlasNdError, AtlasNdResult};
pub use iter::ArrayViewIter;
pub use stride::{compute_strides, element_count};
pub use traits::Numeric;
pub use view::ArrayView;
