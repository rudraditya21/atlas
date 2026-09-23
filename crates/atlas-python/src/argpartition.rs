use pyo3::prelude::*;

use crate::{array, gil, partition_ops::normalize_kths, python_dtype::with_dtype};

pub(crate) fn argpartition(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    kth: &Bound<'_, PyAny>,
    axis: Option<i64>,
) -> PyResult<Py<PyAny>> {
    let dtype = array::source_dtype(py, value)?;
    let kths =
        if let Ok(kth) = kth.extract::<i64>() { vec![kth] } else { kth.extract::<Vec<i64>>()? };

    with_dtype!(
        dtype,
        all | T | {
            let array = array::from_numpy(array::readonly_from_python::<T>(py, value)?)?;
            let result = gil::without_gil(py, move || {
                if let Some(axis) = axis {
                    let kths = normalize_kths(array.shape(), &kths, axis)?;
                    array.argpartition_many(&kths, axis)
                } else {
                    let array = array.flatten();
                    let kths = normalize_kths(array.shape(), &kths, 0)?;
                    array.argpartition_many(&kths, 0)
                }
            })
            .map_err(|error| crate::error::ndarray(py, error))?;
            Ok(array::to_numpy_owned(py, result)?.into_any().unbind())
        }
    )
}
