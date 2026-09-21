use atlas_ndarray::{ArrayElement, NDArray};
use numpy::Element;
use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{array, gil};

macro_rules! with_array {
    ($py:expr, $value:expr, |$array:ident| $body:expr) => {{
        array::require_numpy_array($py, $value)?;
        let dtype: String = $value.getattr("dtype")?.getattr("name")?.extract()?;
        match dtype.as_str() {
            "bool" => {
                let $array = array::from_numpy(array::readonly_from_python::<bool>($py, $value)?)?;
                $body
            }
            "int8" => {
                let $array = array::from_numpy(array::readonly_from_python::<i8>($py, $value)?)?;
                $body
            }
            "int16" => {
                let $array = array::from_numpy(array::readonly_from_python::<i16>($py, $value)?)?;
                $body
            }
            "int32" => {
                let $array = array::from_numpy(array::readonly_from_python::<i32>($py, $value)?)?;
                $body
            }
            "int64" => {
                let $array = array::from_numpy(array::readonly_from_python::<i64>($py, $value)?)?;
                $body
            }
            "uint8" => {
                let $array = array::from_numpy(array::readonly_from_python::<u8>($py, $value)?)?;
                $body
            }
            "uint16" => {
                let $array = array::from_numpy(array::readonly_from_python::<u16>($py, $value)?)?;
                $body
            }
            "uint32" => {
                let $array = array::from_numpy(array::readonly_from_python::<u32>($py, $value)?)?;
                $body
            }
            "uint64" => {
                let $array = array::from_numpy(array::readonly_from_python::<u64>($py, $value)?)?;
                $body
            }
            "float32" => {
                let $array = array::from_numpy(array::readonly_from_python::<f32>($py, $value)?)?;
                $body
            }
            "float64" => {
                let $array = array::from_numpy(array::readonly_from_python::<f64>($py, $value)?)?;
                $body
            }
            _ => Err(PyTypeError::new_err(format!("unsupported NumPy dtype {dtype}"))),
        }
    }};
}

pub(crate) fn reshape(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    shape: Vec<usize>,
) -> PyResult<Py<PyAny>> {
    with_array!(py, value, |array| reshape_array(py, array, shape))
}

pub(crate) fn transpose(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    with_array!(py, value, |array| transpose_array(py, array))
}

fn reshape_array<T>(py: Python<'_>, array: NDArray<T>, shape: Vec<usize>) -> PyResult<Py<PyAny>>
where
    T: ArrayElement + Element,
{
    let array = gil::without_gil(py, move || array.reshape(shape).map(|view| view.to_owned()))
        .map_err(|error| crate::error::ndarray(py, error))?;

    Ok(array::to_numpy_owned(py, array)?.into_any().unbind())
}

fn transpose_array<T>(py: Python<'_>, array: NDArray<T>) -> PyResult<Py<PyAny>>
where
    T: ArrayElement + Element,
{
    let array = gil::without_gil(py, move || array.view().transpose().to_owned());

    Ok(array::to_numpy_owned(py, array)?.into_any().unbind())
}
