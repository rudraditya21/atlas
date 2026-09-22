use atlas_ndarray::AtlasNdError;
use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{array, gil};

pub(crate) fn tile(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    repetitions: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let repetitions = repetition_values(py, repetitions)?;
    array::require_numpy_array(py, value)?;
    let dtype: String = value.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let array = array::from_numpy(array::readonly_from_python::<$ty>(py, value)?)?;
            let result = gil::without_gil(py, move || array.tile(&repetitions))
                .map_err(|error| crate::error::ndarray(py, error))?;
            Ok(array::to_numpy_owned(py, result)?.into_any().unbind())
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
}

fn repetition_values(py: Python<'_>, repetitions: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let repetitions = repetitions
        .extract::<i64>()
        .map(|value| vec![value])
        .or_else(|_| repetitions.extract::<Vec<i64>>())?;

    repetitions
        .into_iter()
        .map(|value| {
            usize::try_from(value).map_err(|_| {
                crate::error::ndarray(
                    py,
                    AtlasNdError::InvalidArgument {
                        op: "tile",
                        reason: "repetitions must be nonnegative",
                    },
                )
            })
        })
        .collect()
}
