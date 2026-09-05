//! Arrow interoperability boundary for Atlas.
//!
//! This crate owns all optional Arrow and Polars integration. `atlas-ndarray` deliberately has
//! no dependency on this crate, Arrow, or Polars, so its core array representation remains
//! independent of external columnar-memory ecosystems.

#![forbid(unsafe_code)]

mod dtype;
mod error;
mod primitive;
mod record_batch;

pub use dtype::InterchangeDType;
pub use error::{AtlasArrowError, AtlasArrowResult};
pub use primitive::{ArrowPrimitive, from_arrow_primitive, to_arrow_primitive};
pub use record_batch::to_arrow_record_batch;
