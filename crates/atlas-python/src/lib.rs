//! Native Python bindings for Atlas.
//!
//! This crate owns the Python extension boundary. Numerical implementations remain in the
//! Atlas library crates and will be registered here incrementally.

use pyo3::prelude::*;

mod arithmetic;
mod array;
mod casting;
mod constructors;
mod error;
mod gil;
mod logical;
mod metadata;
#[path = "dtype.rs"]
mod python_dtype;
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
    module.add_function(wrap_pyfunction!(astype, module)?)?;
    module.add_function(wrap_pyfunction!(shape, module)?)?;
    module.add_function(wrap_pyfunction!(ndim, module)?)?;
    module.add_function(wrap_pyfunction!(size, module)?)?;
    module.add_function(wrap_pyfunction!(dtype, module)?)?;
    module.add_function(wrap_pyfunction!(add, module)?)?;
    module.add_function(wrap_pyfunction!(subtract, module)?)?;
    module.add_function(wrap_pyfunction!(multiply, module)?)?;
    module.add_function(wrap_pyfunction!(divide, module)?)?;
    module.add_function(wrap_pyfunction!(equal, module)?)?;
    module.add_function(wrap_pyfunction!(not_equal, module)?)?;
    module.add_function(wrap_pyfunction!(less, module)?)?;
    module.add_function(wrap_pyfunction!(less_equal, module)?)?;
    module.add_function(wrap_pyfunction!(greater, module)?)?;
    module.add_function(wrap_pyfunction!(greater_equal, module)?)?;
    module.add_function(wrap_pyfunction!(select, module)?)?;
    module.add_function(wrap_pyfunction!(count_true, module)?)?;
    module.add_function(wrap_pyfunction!(nonzero, module)?)?;
    module.add_function(wrap_pyfunction!(masked_fill, module)?)?;

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

#[pyfunction]
fn astype(py: Python<'_>, value: &Bound<'_, PyAny>, dtype: &str) -> PyResult<Py<PyAny>> {
    casting::astype(py, value, dtype)
}

#[pyfunction]
fn shape(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    metadata::shape(py, value)
}

#[pyfunction]
fn ndim(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<usize> {
    metadata::ndim(py, value)
}

#[pyfunction]
fn size(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<usize> {
    metadata::size(py, value)
}

#[pyfunction]
fn dtype(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    metadata::dtype(py, value)
}

#[pyfunction]
fn add(py: Python<'_>, lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    arithmetic::add(py, lhs, rhs)
}

#[pyfunction]
fn subtract(py: Python<'_>, lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    arithmetic::subtract(py, lhs, rhs)
}

#[pyfunction]
fn multiply(py: Python<'_>, lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    arithmetic::multiply(py, lhs, rhs)
}

#[pyfunction]
fn divide(py: Python<'_>, lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    arithmetic::divide(py, lhs, rhs)
}

#[pyfunction]
fn equal(py: Python<'_>, lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    logical::equal(py, lhs, rhs)
}

#[pyfunction]
fn not_equal(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    logical::not_equal(py, lhs, rhs)
}

#[pyfunction]
fn less(py: Python<'_>, lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    logical::less(py, lhs, rhs)
}

#[pyfunction]
fn less_equal(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    logical::less_equal(py, lhs, rhs)
}

#[pyfunction]
fn greater(py: Python<'_>, lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    logical::greater(py, lhs, rhs)
}

#[pyfunction]
fn greater_equal(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    logical::greater_equal(py, lhs, rhs)
}

#[pyfunction]
fn select(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    mask: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    logical::select(py, value, mask)
}

#[pyfunction]
fn count_true(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<usize> {
    logical::count_true(py, value)
}

#[pyfunction]
fn nonzero(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    logical::nonzero(py, value)
}

#[pyfunction]
fn masked_fill(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    mask: &Bound<'_, PyAny>,
    fill: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    logical::masked_fill(py, value, mask, fill)
}
