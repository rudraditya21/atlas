use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{array, gil};

#[derive(Clone, Copy)]
enum SolveOperation {
    Regular,
    Transpose,
}

pub(crate) fn solve(
    py: Python<'_>,
    matrix: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    solve_with(py, matrix, rhs, SolveOperation::Regular)
}

pub(crate) fn solve_transpose(
    py: Python<'_>,
    matrix: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    solve_with(py, matrix, rhs, SolveOperation::Transpose)
}

fn solve_with(
    py: Python<'_>,
    matrix: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    operation: SolveOperation,
) -> PyResult<Py<PyAny>> {
    array::require_numpy_array(py, matrix)?;
    array::require_numpy_array(py, rhs)?;
    let dtype: String = matrix.getattr("dtype")?.getattr("name")?.extract()?;
    let operation_name = match operation {
        SolveOperation::Regular => "solve",
        SolveOperation::Transpose => "solve_transpose",
    };

    macro_rules! apply {
        ($ty:ty) => {{
            let matrix = array::from_numpy(array::readonly_from_python::<$ty>(py, matrix)?)?;
            let rhs = array::from_numpy(array::readonly_from_python::<$ty>(py, rhs)?)?;
            let result = gil::without_gil(py, move || match operation {
                SolveOperation::Regular => atlas_linalg::solve(&matrix, &rhs),
                SolveOperation::Transpose => atlas_linalg::solve_transpose(&matrix, &rhs),
            })
            .map_err(|error| crate::error::linalg(py, error))?;
            Ok(array::to_numpy_owned(py, result)?.into_any().unbind())
        }};
    }

    match dtype.as_str() {
        "float32" => apply!(f32),
        "float64" => apply!(f64),
        _ => Err(PyTypeError::new_err(format!(
            "{operation_name} requires float32 or float64 NumPy arrays"
        ))),
    }
}
