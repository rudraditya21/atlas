//! Support Python bindings.

use pyo3::{prelude::*, wrap_pyfunction};

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(version, module)?)?;

    #[cfg(feature = "test-support")]
    crate::test_support::register(module)?;

    Ok(())
}
