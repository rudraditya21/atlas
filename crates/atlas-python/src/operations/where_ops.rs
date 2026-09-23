use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{array, gil};

pub(crate) fn where_(
    py: Python<'_>,
    condition: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let condition = array::from_numpy(array::readonly_from_python::<bool>(py, condition)?)?;
    let source = if array::is_numpy_array(py, x)? {
        x
    } else if array::is_numpy_array(py, y)? {
        y
    } else {
        return Err(PyTypeError::new_err("where requires at least one NumPy array operand"));
    };
    let dtype: String = source.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let result = if array::is_numpy_array(py, x)? {
                let x = array::from_numpy(array::readonly_from_python::<$ty>(py, x)?)?;
                if array::is_numpy_array(py, y)? {
                    let y = array::from_numpy(array::readonly_from_python::<$ty>(py, y)?)?;
                    gil::without_gil(py, move || condition.r#where(&x, &y))
                } else {
                    let y = y.extract::<$ty>()?;
                    gil::without_gil(py, move || condition.r#where(&x, y))
                }
            } else {
                let x = x.extract::<$ty>()?;
                let y = array::from_numpy(array::readonly_from_python::<$ty>(py, y)?)?;
                gil::without_gil(py, move || condition.r#where(x, &y))
            };
            let result = result.map_err(|error| crate::error::ndarray(py, error))?;
            Ok(array::to_numpy_owned(py, result)?.into_any().unbind())
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
        _ => Err(PyTypeError::new_err("where requires supported matching numeric NumPy dtypes")),
    }
}
