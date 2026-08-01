pub mod arithmetic;
pub mod array;
pub mod constructors;
pub mod error;
pub mod indexing;
pub mod iter;
pub mod reshape;
pub mod slicing;
pub mod stride;
pub mod traits;
pub mod transpose;
pub mod view;

pub use error::{AtlasNdError, AtlasNdResult};
