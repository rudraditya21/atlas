//! Shared native binding support modules.

pub(crate) mod arrays;
pub(crate) mod dtypes;
pub(crate) mod errors;
pub(crate) mod gil;
pub(crate) mod metadata;
pub(crate) mod results;
#[cfg(feature = "test-support")]
pub(crate) mod scalar;
#[cfg(feature = "test-support")]
pub(crate) mod test_support;

use pyo3::{prelude::*, wrap_pyfunction};

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(version, module)?)?;

    #[cfg(feature = "test-support")]
    test_support::register(module)?;

    Ok(())
}
