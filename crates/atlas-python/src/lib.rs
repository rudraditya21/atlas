//! Native Python bindings for Atlas.
//!
//! This crate owns the Python extension boundary. Numerical implementations remain in the
//! Atlas library crates and will be registered here incrementally.

use pyo3::prelude::*;

mod array;
mod constructors;
#[cfg(feature = "test-support")]
mod error;
#[cfg(feature = "test-support")]
mod gil;
#[cfg(feature = "test-support")]
mod scalar;
#[cfg(feature = "test-support")]
mod test_support;

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add_function(wrap_pyfunction!(asarray, module)?)?;
    module.add_function(wrap_pyfunction!(zeros, module)?)?;
    module.add_function(wrap_pyfunction!(ones, module)?)?;
    module.add_function(wrap_pyfunction!(full, module)?)?;
    module.add_function(wrap_pyfunction!(arange, module)?)?;

    #[cfg(feature = "test-support")]
    test_support::register(module)?;

    Ok(())
}

#[pyfunction(signature = (value, dtype = None))]
fn asarray(py: Python<'_>, value: &Bound<'_, PyAny>, dtype: Option<&str>) -> PyResult<Py<PyAny>> {
    constructors::asarray(py, value, dtype)
}

#[pyfunction(signature = (shape, dtype = None))]
fn zeros(py: Python<'_>, shape: Vec<usize>, dtype: Option<&str>) -> PyResult<Py<PyAny>> {
    constructors::zeros(py, shape, dtype)
}

#[pyfunction(signature = (shape, dtype = None))]
fn ones(py: Python<'_>, shape: Vec<usize>, dtype: Option<&str>) -> PyResult<Py<PyAny>> {
    constructors::ones(py, shape, dtype)
}

#[pyfunction(signature = (shape, value, dtype = None))]
fn full(
    py: Python<'_>,
    shape: Vec<usize>,
    value: &Bound<'_, PyAny>,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    constructors::full(py, shape, value, dtype)
}

#[pyfunction(signature = (start, stop = None, step = None, dtype = None))]
fn arange(
    py: Python<'_>,
    start: f64,
    stop: Option<f64>,
    step: Option<f64>,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    constructors::arange(py, start, stop, step, dtype)
}
