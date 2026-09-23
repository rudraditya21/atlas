use atlas_ndarray::{ArrayElement, NDArray, RuntimeScalar};
use numpy::Element;
use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{array, gil, python_dtype::DType};

pub(crate) fn astype(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    dtype: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let target = DType::parse(py, Some(dtype))?;
    array::require_numpy_array(py, value)?;
    let source: String = value.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! cast_from {
        ($ty:ty) => {{
            let array = array::from_numpy(array::readonly_from_python::<$ty>(py, value)?)?;
            match target {
                DType::Bool => cast::<$ty, bool>(py, array),
                DType::Int8 => cast::<$ty, i8>(py, array),
                DType::Int16 => cast::<$ty, i16>(py, array),
                DType::Int32 => cast::<$ty, i32>(py, array),
                DType::Int64 => cast::<$ty, i64>(py, array),
                DType::UInt8 => cast::<$ty, u8>(py, array),
                DType::UInt16 => cast::<$ty, u16>(py, array),
                DType::UInt32 => cast::<$ty, u32>(py, array),
                DType::UInt64 => cast::<$ty, u64>(py, array),
                DType::Float32 => cast::<$ty, f32>(py, array),
                DType::Float64 => cast::<$ty, f64>(py, array),
            }
        }};
    }

    match source.as_str() {
        "bool" => cast_from!(bool),
        "int8" => cast_from!(i8),
        "int16" => cast_from!(i16),
        "int32" => cast_from!(i32),
        "int64" => cast_from!(i64),
        "uint8" => cast_from!(u8),
        "uint16" => cast_from!(u16),
        "uint32" => cast_from!(u32),
        "uint64" => cast_from!(u64),
        "float32" => cast_from!(f32),
        "float64" => cast_from!(f64),
        _ => Err(PyTypeError::new_err(format!("unsupported NumPy dtype {source}"))),
    }
}

fn cast<T, U>(py: Python<'_>, array: NDArray<T>) -> PyResult<Py<PyAny>>
where
    T: ArrayElement + RuntimeScalar + Element,
    U: ArrayElement + RuntimeScalar + Element,
{
    let array = gil::without_gil(py, move || array.astype::<U>())
        .map_err(|error| crate::error::ndarray(py, error))?;

    Ok(array::to_numpy_owned(py, array)?.into_any().unbind())
}
