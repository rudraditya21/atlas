use pyo3::prelude::*;

use crate::array;

pub(crate) fn shape(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    attribute(py, value, "shape")
}

pub(crate) fn ndim(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<usize> {
    attribute(py, value, "ndim")?.bind(py).extract()
}

pub(crate) fn size(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<usize> {
    attribute(py, value, "size")?.bind(py).extract()
}

pub(crate) fn dtype(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    attribute(py, value, "dtype")
}

fn attribute(py: Python<'_>, value: &Bound<'_, PyAny>, name: &str) -> PyResult<Py<PyAny>> {
    array::require_numpy_array(py, value)?;

    Ok(value.getattr(name)?.unbind())
}
