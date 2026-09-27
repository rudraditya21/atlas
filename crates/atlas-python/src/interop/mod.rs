//! Interop Python binding modules.

mod arrow;

use pyo3::{prelude::*, wrap_pyfunction};

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(to_arrow_primitive, module)?)?;
    module.add_function(wrap_pyfunction!(from_arrow_primitive, module)?)?;
    module.add_function(wrap_pyfunction!(to_arrow_record_batch, module)?)?;
    module.add_function(wrap_pyfunction!(from_arrow_record_batch, module)?)?;
    Ok(())
}

#[pyfunction]
fn to_arrow_primitive(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> crate::support::results::PyObjectResult {
    arrow::to_arrow_primitive(py, value)
}

#[pyfunction]
fn from_arrow_primitive(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> crate::support::results::PyObjectResult {
    arrow::from_arrow_primitive(py, value)
}

#[pyfunction]
fn to_arrow_record_batch(
    py: Python<'_>,
    matrix: &Bound<'_, PyAny>,
    column_names: Vec<String>,
) -> crate::support::results::PyObjectResult {
    arrow::to_arrow_record_batch(py, matrix, column_names)
}

#[pyfunction]
fn from_arrow_record_batch(
    py: Python<'_>,
    batch: &Bound<'_, PyAny>,
) -> crate::support::results::PyObjectResult {
    arrow::from_arrow_record_batch(py, batch)
}
