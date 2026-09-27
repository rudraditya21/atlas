use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{array, gil};

pub(crate) fn symmetric_eigendecomposition(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    array::require_numpy_array(py, value)?;
    let dtype: String = value.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let value = array::from_numpy(array::readonly_from_python::<$ty>(py, value)?)?;
            let decomposition =
                gil::without_gil(py, move || atlas_linalg::symmetric_eigendecomposition(&value))
                    .map_err(|error| crate::error::linalg(py, error))?;
            Ok((
                array::to_numpy_owned(py, decomposition.eigenvalues().clone())?.into_any().unbind(),
                array::to_numpy_owned(py, decomposition.eigenvectors().clone())?
                    .into_any()
                    .unbind(),
            ))
        }};
    }

    match dtype.as_str() {
        "float32" => apply!(f32),
        "float64" => apply!(f64),
        _ => Err(PyTypeError::new_err(
            "symmetric_eigendecomposition requires a float32 or float64 NumPy array",
        )),
    }
}
