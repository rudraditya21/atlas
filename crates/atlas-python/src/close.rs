use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{array, gil};

pub(crate) fn allclose(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> PyResult<bool> {
    array::require_numpy_array(py, lhs)?;
    let dtype: String = lhs.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! compare {
        ($ty:ty) => {{
            let lhs = array::from_numpy(array::readonly_from_python::<$ty>(py, lhs)?)?;
            let rhs = array::from_numpy(array::readonly_from_python::<$ty>(py, rhs)?)?;
            gil::without_gil(py, move || {
                atlas_ndarray::allclose(&lhs, &rhs, rtol as $ty, atol as $ty, equal_nan)
            })
            .map_err(|error| crate::error::ndarray(py, error))
        }};
    }

    match dtype.as_str() {
        "float32" => compare!(f32),
        "float64" => compare!(f64),
        _ => Err(PyTypeError::new_err("allclose requires a float32 or float64 NumPy array")),
    }
}
