//! Native Python bindings for Atlas.
//!
//! This crate owns the Python extension boundary. Numerical implementations remain in the
//! Atlas library crates and will be registered here incrementally.

use pyo3::prelude::*;

#[path = "bindings/arrays.rs"]
mod array_bindings;
#[path = "bindings/support.rs"]
mod support_bindings;

#[path = "support/array.rs"]
mod array;
mod constructors;
#[path = "support/error.rs"]
mod error;
#[path = "support/gil.rs"]
mod gil;
mod indexing;
mod interop;
mod linalg;
mod manipulation;
mod ml;
mod operations;
#[path = "support/dtype.rs"]
mod python_dtype;
mod random;
mod reductions;
#[path = "support/results.rs"]
mod results;
#[cfg(feature = "test-support")]
#[path = "support/scalar.rs"]
mod scalar;
mod statistics;
#[cfg(feature = "test-support")]
#[path = "support/test_support.rs"]
mod test_support;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    support_bindings::register(module)?;
    random::register(module)?;
    constructors::register(module)?;
    operations::register(module)?;
    reductions::register(module)?;
    indexing::register(module)?;
    manipulation::register(module)?;
    array_bindings::register(module)?;
    linalg::register(module)?;
    statistics::register(module)?;
    ml::register(module)?;
    interop::register(module)?;
    Ok(())
}
