use pyo3::prelude::*;

use crate::{array, gil, python_dtype::with_dtype};

pub(crate) fn take(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    indices: Vec<i64>,
    axis: i64,
) -> PyResult<Py<PyAny>> {
    let dtype = array::source_dtype(py, value)?;

    with_dtype!(
        dtype,
        all | T | {
            let value = array::from_numpy(array::readonly_from_python::<T>(py, value)?)?;
            let result = gil::without_gil(py, move || value.take(&indices, axis))
                .map_err(|error| crate::error::ndarray(py, error))?;
            Ok(array::to_numpy_owned(py, result)?.into_any().unbind())
        }
    )
}
