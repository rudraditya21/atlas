//! Shared native binding support modules.

pub(crate) mod arrays;
pub(crate) mod dtypes;
pub(crate) mod errors;
pub(crate) mod gil;
pub(crate) mod metadata;
mod registration;
pub(crate) mod results;
#[cfg(feature = "test-support")]
pub(crate) mod scalar;
#[cfg(feature = "test-support")]
pub(crate) mod test_support;

use pyo3::{prelude::*, types::PyDict, wrap_pyfunction};

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction(name = "_backend_config")]
fn backend_config<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let configuration = atlas_blas::backend_configuration();
    let result = PyDict::new(py);
    result.set_item("provider", configuration.provider_name())?;
    result.set_item("integer_width", configuration.integer_width())?;
    result.set_item("thread_control_available", configuration.thread_control_available())?;

    Ok(result)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add_function(wrap_pyfunction!(backend_config, module)?)?;

    #[cfg(feature = "test-support")]
    test_support::register(module)?;

    Ok(())
}
