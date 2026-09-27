//! Native Python bindings for Atlas.
//!
//! This crate owns the Python extension boundary. Numerical implementations remain in the
//! Atlas library crates and will be registered here incrementally.

use pyo3::prelude::*;

mod constructors;
mod indexing;
mod interop;
mod linalg;
mod manipulation;
mod ml;
mod operations;
mod random;
mod reductions;
mod statistics;
mod support;

#[cfg(feature = "test-support")]
pub(crate) use support::scalar;
pub(crate) use support::{array, error, gil, python_dtype, results};

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    support::register(module)?;
    random::register(module)?;
    constructors::register(module)?;
    operations::register(module)?;
    reductions::register(module)?;
    indexing::register(module)?;
    manipulation::register(module)?;
    linalg::register(module)?;
    statistics::register(module)?;
    ml::register(module)?;
    interop::register(module)?;
    Ok(())
}
