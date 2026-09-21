use atlas_linalg::MatrixNorm;
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};

use crate::{array, gil};

pub(crate) fn matrix_norm(py: Python<'_>, value: &Bound<'_, PyAny>, order: &str) -> PyResult<f64> {
    let order = parse_order(order)?;
    array::require_numpy_array(py, value)?;
    let dtype: String = value.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let value = array::from_numpy(array::readonly_from_python::<$ty>(py, value)?)?;
            gil::without_gil(py, move || atlas_linalg::matrix_norm(&value, order))
                .map_err(|error| crate::error::linalg(py, error))
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
        "bool" => Err(PyTypeError::new_err("matrix_norm does not support bool dtype")),
        _ => Err(PyTypeError::new_err(format!("unsupported NumPy dtype {dtype}"))),
    }
}

fn parse_order(order: &str) -> PyResult<MatrixNorm> {
    match order {
        "fro" => Ok(MatrixNorm::Frobenius),
        "l1" => Ok(MatrixNorm::L1),
        "inf" => Ok(MatrixNorm::Infinity),
        _ => Err(PyValueError::new_err("matrix norm order must be 'fro', 'l1', or 'inf'")),
    }
}
