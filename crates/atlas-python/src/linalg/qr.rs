use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::support::{arrays as array, gil};

pub(crate) fn qr(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    array::require_numpy_array(py, value)?;
    let dtype: String = value.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let value = array::from_numpy(array::readonly_from_python::<$ty>(py, value)?)?;
            let factor = gil::without_gil(py, move || atlas_linalg::qr(&value))
                .map_err(|error| crate::support::errors::linalg(py, error))?;
            Ok((
                array::to_numpy_owned(py, factor.q().clone())?.into_any().unbind(),
                array::to_numpy_owned(py, factor.r().clone())?.into_any().unbind(),
            ))
        }};
    }

    match dtype.as_str() {
        "float32" => apply!(f32),
        "float64" => apply!(f64),
        _ => Err(PyTypeError::new_err("qr requires a float32 or float64 NumPy array")),
    }
}

pub(crate) fn least_squares(
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
            let result = gil::without_gil(py, move || atlas_linalg::least_squares(&matrix, &rhs))
                .map_err(|error| crate::support::errors::linalg(py, error))?;
            Ok(array::to_numpy_owned(py, result)?.into_any().unbind())
        }};
    }

    match dtype.as_str() {
        "float32" => apply!(f32),
        "float64" => apply!(f64),
        _ => Err(PyTypeError::new_err("least_squares requires float32 or float64 NumPy arrays")),
    }
}

pub(crate) fn matrix_rank(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<usize> {
    array::require_numpy_array(py, value)?;
    let dtype: String = value.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let value = array::from_numpy(array::readonly_from_python::<$ty>(py, value)?)?;
            gil::without_gil(py, move || atlas_linalg::matrix_rank(&value))
                .map_err(|error| crate::support::errors::linalg(py, error))
        }};
    }

    match dtype.as_str() {
        "float32" => apply!(f32),
        "float64" => apply!(f64),
        _ => Err(PyTypeError::new_err("matrix_rank requires a float32 or float64 NumPy array")),
    }
}
