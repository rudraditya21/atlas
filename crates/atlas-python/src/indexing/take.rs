use pyo3::prelude::*;

use crate::support::{arrays as array, dtypes::with_dtype, gil};

pub(crate) fn take(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    indices: Vec<i64>,
    axis: Option<i64>,
) -> PyResult<Py<PyAny>> {
    let dtype = array::source_dtype(py, value)?;

    with_dtype!(
        dtype,
        all | T | {
            let value = array::from_numpy(array::readonly_from_python::<T>(py, value)?)?;
            let result = gil::without_gil(py, move || match axis {
                Some(axis) => value.take(&indices, axis),
                None => value.flatten().take(&indices, 0),
            })
            .map_err(|error| crate::support::errors::ndarray(py, error))?;
            Ok(array::to_numpy_owned(py, result)?.into_any().unbind())
        }
    )
}
