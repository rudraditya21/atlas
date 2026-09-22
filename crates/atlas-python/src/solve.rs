use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{array, gil};

pub(crate) fn solve(
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
            let result = gil::without_gil(py, move || atlas_linalg::solve(&matrix, &rhs))
                .map_err(|error| crate::error::linalg(py, error))?;
            Ok(array::to_numpy_owned(py, result)?.into_any().unbind())
        }};
    }

    match dtype.as_str() {
        "float32" => apply!(f32),
        "float64" => apply!(f64),
        _ => Err(PyTypeError::new_err("solve requires float32 or float64 NumPy arrays")),
    }
}
