//! Constructor Python binding modules.

mod casting;
#[path = "constructors.rs"]
mod functions;

use pyo3::{prelude::*, wrap_pyfunction};

use crate::support::metadata;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(asarray, module)?)?;
    module.add_function(wrap_pyfunction!(zeros, module)?)?;
    module.add_function(wrap_pyfunction!(ones, module)?)?;
    module.add_function(wrap_pyfunction!(eye, module)?)?;
    module.add_function(wrap_pyfunction!(identity, module)?)?;
    module.add_function(wrap_pyfunction!(full, module)?)?;
    module.add_function(wrap_pyfunction!(arange, module)?)?;
    module.add_function(wrap_pyfunction!(linspace, module)?)?;
    module.add_function(wrap_pyfunction!(astype, module)?)?;
    module.add_function(wrap_pyfunction!(shape, module)?)?;
    module.add_function(wrap_pyfunction!(ndim, module)?)?;
    module.add_function(wrap_pyfunction!(size, module)?)?;
    module.add_function(wrap_pyfunction!(dtype, module)?)?;
    Ok(())
}

#[pyfunction(signature = (value, dtype = None))]
fn asarray(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    functions::asarray(py, value, dtype)
}

#[pyfunction(signature = (shape, dtype = None))]
fn zeros(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    functions::zeros(py, shape, dtype)
}

#[pyfunction(signature = (shape, dtype = None))]
fn ones(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    functions::ones(py, shape, dtype)
}

#[pyfunction(signature = (rows, columns = None, dtype = None))]
fn eye(
    py: Python<'_>,
    rows: usize,
    columns: Option<usize>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    functions::eye(py, rows, columns, dtype)
}

#[pyfunction(signature = (size, dtype = None))]
fn identity(py: Python<'_>, size: usize, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    functions::identity(py, size, dtype)
}

#[pyfunction(signature = (shape, fill_value, dtype = None))]
fn full(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    fill_value: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    functions::full(py, shape, fill_value, dtype)
}

#[pyfunction(signature = (start, stop = None, step = None, dtype = None))]
fn arange(
    py: Python<'_>,
    start: &Bound<'_, PyAny>,
    stop: Option<&Bound<'_, PyAny>>,
    step: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    functions::arange(py, start, stop, step, dtype)
}

#[pyfunction(signature = (start, stop, num, dtype = None, endpoint = true))]
fn linspace(
    py: Python<'_>,
    start: f64,
    stop: f64,
    num: usize,
    dtype: Option<&Bound<'_, PyAny>>,
    endpoint: bool,
) -> PyResult<Py<PyAny>> {
    functions::linspace(py, start, stop, num, dtype, endpoint)
}

#[pyfunction(signature = (value, dtype, copy = true))]
fn astype(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    dtype: &Bound<'_, PyAny>,
    copy: bool,
) -> PyResult<Py<PyAny>> {
    casting::astype(py, value, dtype, copy)
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
