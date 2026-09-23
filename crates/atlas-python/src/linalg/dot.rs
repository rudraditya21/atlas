use pyo3::{IntoPyObjectExt, exceptions::PyTypeError, prelude::*};

use crate::{array, gil};

pub(crate) fn dot(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    array::require_numpy_array(py, lhs)?;
    let dtype: String = lhs.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let lhs = array::from_numpy(array::readonly_from_python::<$ty>(py, lhs)?)?;
            let rhs = array::from_numpy(array::readonly_from_python::<$ty>(py, rhs)?)?;
            match gil::without_gil(py, move || atlas_linalg::dot(&lhs, &rhs))
                .map_err(|error| crate::error::linalg(py, error))?
            {
                atlas_linalg::DotOutput::Scalar(value) => value.into_py_any(py),
                atlas_linalg::DotOutput::Array(array) => {
                    Ok(array::to_numpy_owned(py, array)?.into_any().unbind())
                }
            }
        }};
    }

    match dtype.as_str() {
        "int8" => apply!(i8),
        "int16" => apply!(i16),
        "int32" => apply!(i32),
        "int64" => apply!(i64),
        "uint8" => apply!(u8),
        "uint16" => apply!(u16),
        "uint32" => apply!(u32),
        "uint64" => apply!(u64),
        "float32" => apply!(f32),
        "float64" => apply!(f64),
        "bool" => Err(PyTypeError::new_err("dot does not support bool dtype")),
        _ => Err(PyTypeError::new_err(format!("unsupported NumPy dtype {dtype}"))),
    }
}
