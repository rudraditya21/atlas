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

pub mod arithmetic;
pub mod array;
pub mod axis;
pub mod broadcast;
pub mod constructors;
pub mod error;
pub mod indexing;
pub mod iter;
pub mod reduction;
pub mod reshape;
pub mod slicing;
pub mod stride;
pub mod traits;
pub mod transpose;
pub mod view;
pub(crate) mod traversal;

pub use array::NDArray;
pub use axis::AxisIndex;
pub use broadcast::{
    broadcast_pair, broadcast_shape, broadcast_strides, contiguous_broadcast_metadata,
    BroadcastMetadata,
};
pub use error::{AtlasNdError, AtlasNdResult};
pub use traits::Numeric;
pub use view::ArrayView;
