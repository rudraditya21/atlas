use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::support::{arrays as array, gil};

pub(crate) fn cholesky(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    array::require_numpy_array(py, value)?;
    let dtype: String = value.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let value = array::from_numpy(array::readonly_from_python::<$ty>(py, value)?)?;
            let result = gil::without_gil(py, move || {
                atlas_linalg::cholesky(&value).map(|factor| factor.l().clone())
            })
            .map_err(|error| crate::support::errors::linalg(py, error))?;
            Ok(array::to_numpy_owned(py, result)?.into_any().unbind())
        }};
    }

    match dtype.as_str() {
        "float32" => apply!(f32),
        "float64" => apply!(f64),
        _ => Err(PyTypeError::new_err("cholesky requires a float32 or float64 NumPy array")),
    }
}

pub(crate) fn solve_spd(
    py: Python<'_>,
    matrix: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    array::require_numpy_array(py, matrix)?;
    array::require_numpy_array(py, rhs)?;
    let dtype: String = matrix.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let matrix = array::from_numpy(array::readonly_from_python::<$ty>(py, matrix)?)?;
            let rhs = array::from_numpy(array::readonly_from_python::<$ty>(py, rhs)?)?;
            let result = gil::without_gil(py, move || atlas_linalg::solve_spd(&matrix, &rhs))
                .map_err(|error| crate::support::errors::linalg(py, error))?;
            Ok(array::to_numpy_owned(py, result)?.into_any().unbind())
        }};
    }

    match dtype.as_str() {
        "float32" => apply!(f32),
        "float64" => apply!(f64),
        _ => Err(PyTypeError::new_err("solve_spd requires float32 or float64 NumPy arrays")),
    }
}
