use pyo3::prelude::*;

use crate::{array, gil, python_dtype::with_dtype};

pub(crate) fn argsort(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    axis: Option<i64>,
) -> PyResult<Py<PyAny>> {
    let dtype = array::source_dtype(py, value)?;

    with_dtype!(
        dtype,
        all | T | {
            let array = array::from_numpy(array::readonly_from_python::<T>(py, value)?)?;
            let result = gil::without_gil(py, move || match axis {
                Some(axis) => array.argsort(axis),
                None => array.flatten().argsort(0),
            })
            .map_err(|error| crate::error::ndarray(py, error))?;
            Ok(array::to_numpy_owned(py, result)?.into_any().unbind())
        }
    )
}
