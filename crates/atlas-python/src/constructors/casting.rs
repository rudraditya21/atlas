use atlas_ndarray::{ArrayElement, NDArray, RuntimeScalar};
use numpy::Element;
use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{
    array, gil,
    python_dtype::{DType, with_dtype},
};

pub(crate) fn astype(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    dtype: &Bound<'_, PyAny>,
    copy: bool,
) -> PyResult<Py<PyAny>> {
    let target = DType::parse(py, Some(dtype))?;
    let source_dtype = array::source_dtype(py, value)?;
    if !copy && source_dtype == target {
        return Ok(value.clone().unbind());
    }

    let source: String = value.getattr("dtype")?.getattr("name")?.extract()?;

    with_dtype!(
        target,
        all | Target
            | match source.as_str() {
                "bool" => cast_from::<bool, Target>(py, value),
                "int8" => cast_from::<i8, Target>(py, value),
                "int16" => cast_from::<i16, Target>(py, value),
                "int32" => cast_from::<i32, Target>(py, value),
                "int64" => cast_from::<i64, Target>(py, value),
                "uint8" => cast_from::<u8, Target>(py, value),
                "uint16" => cast_from::<u16, Target>(py, value),
                "uint32" => cast_from::<u32, Target>(py, value),
                "uint64" => cast_from::<u64, Target>(py, value),
                "float32" => cast_from::<f32, Target>(py, value),
                "float64" => cast_from::<f64, Target>(py, value),
                _ => Err(PyTypeError::new_err(format!("unsupported NumPy dtype {source}"))),
            }
    )
}

fn cast_from<T, U>(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>>
where
    T: ArrayElement + RuntimeScalar + Element,
    U: ArrayElement + RuntimeScalar + Element,
{
    cast::<T, U>(py, array::from_numpy(array::readonly_from_python(py, value)?)?)
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
