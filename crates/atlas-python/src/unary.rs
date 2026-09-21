use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{array, gil};

macro_rules! numeric_operation {
    ($py:expr, $value:expr, $operation:ident, $($dtype:literal => $ty:ty),+ $(,)?) => {{
        array::require_numpy_array($py, $value)?;
        let dtype: String = $value.getattr("dtype")?.getattr("name")?.extract()?;
        match dtype.as_str() {
            $(
                $dtype => {
                    let array = array::from_numpy(array::readonly_from_python::<$ty>($py, $value)?)?;
                    output($py, gil::without_gil($py, move || array.$operation()))
                }
            )+
            "bool" => Err(PyTypeError::new_err(concat!(stringify!($operation), " does not support bool dtype"))),
            _ => Err(PyTypeError::new_err(format!("unsupported NumPy dtype {dtype}"))),
        }
    }};
}

macro_rules! classify {
    ($py:expr, $value:expr, $operation:ident) => {{
        array::require_numpy_array($py, $value)?;
        let dtype: String = $value.getattr("dtype")?.getattr("name")?.extract()?;
        macro_rules! apply {
            ($ty:ty) => {{
                let array = array::from_numpy(array::readonly_from_python::<$ty>($py, $value)?)?;
                output($py, gil::without_gil($py, move || array.$operation()))
            }};
        }

        match dtype.as_str() {
            "bool" => apply!(bool),
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
            _ => Err(PyTypeError::new_err(format!("unsupported NumPy dtype {dtype}"))),
        }
    }};
}

pub(crate) fn neg(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    numeric_operation!(py, value, neg, "int8" => i8, "int16" => i16, "int32" => i32, "int64" => i64, "float32" => f32, "float64" => f64)
}

pub(crate) fn abs(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    numeric_operation!(py, value, abs, "int8" => i8, "int16" => i16, "int32" => i32, "int64" => i64, "uint8" => u8, "uint16" => u16, "uint32" => u32, "uint64" => u64, "float32" => f32, "float64" => f64)
}

pub(crate) fn sign(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    numeric_operation!(py, value, sign, "int8" => i8, "int16" => i16, "int32" => i32, "int64" => i64, "uint8" => u8, "uint16" => u16, "uint32" => u32, "uint64" => u64, "float32" => f32, "float64" => f64)
}

pub(crate) fn round(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    numeric_operation!(py, value, round, "int8" => i8, "int16" => i16, "int32" => i32, "int64" => i64, "uint8" => u8, "uint16" => u16, "uint32" => u32, "uint64" => u64, "float32" => f32, "float64" => f64)
}

pub(crate) fn isnan(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    classify!(py, value, isnan)
}

pub(crate) fn isinf(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    classify!(py, value, isinf)
}

pub(crate) fn isfinite(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    classify!(py, value, isfinite)
}

fn output<T>(py: Python<'_>, array: atlas_ndarray::NDArray<T>) -> PyResult<Py<PyAny>>
where
    T: atlas_ndarray::ArrayElement + numpy::Element,
{
    Ok(array::to_numpy_owned(py, array)?.into_any().unbind())
}
