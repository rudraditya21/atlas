//! Arrow interoperability boundary for Atlas.
//!
//! This crate owns all optional Arrow and Polars integration. `atlas-ndarray` deliberately has
//! no dependency on this crate, Arrow, or Polars, so its core array representation remains
//! independent of external columnar-memory ecosystems.
//!
//! ## Null handling
//!
//! Atlas ndarrays have no validity bitmap. Conversions from Arrow reject arrays containing null
//! values with [`AtlasArrowError::NullValues`]; nulls are never coerced to a numeric value or
//! `NaN`. Conversions to Arrow always produce non-nullable arrays and record-batch fields.
//!
//! ## Ownership
//!
//! Conversions allocate exactly one destination buffer per converted array or column. Arrow
//! offsets are read through their logical values, while RecordBatches remain independent chunks
//! that callers convert individually.

#![forbid(unsafe_code)]

mod dtype;
mod error;
mod primitive;
mod record_batch;

pub use dtype::InterchangeDType;
pub use error::{AtlasArrowError, AtlasArrowResult};
pub use primitive::{ArrowPrimitive, from_arrow_primitive, to_arrow_primitive};
pub use record_batch::{from_arrow_record_batch, to_arrow_record_batch};
