use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{array, gil, partition_ops::normalize_kths};

pub(crate) fn argpartition(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    kth: &Bound<'_, PyAny>,
    axis: Option<i64>,
) -> PyResult<Py<PyAny>> {
    array::require_numpy_array(py, value)?;
    let dtype: String = value.getattr("dtype")?.getattr("name")?.extract()?;
    let kths =
        if let Ok(kth) = kth.extract::<i64>() { vec![kth] } else { kth.extract::<Vec<i64>>()? };

    macro_rules! apply {
        ($ty:ty) => {{
            let array = array::from_numpy(array::readonly_from_python::<$ty>(py, value)?)?;
            let result = gil::without_gil(py, move || {
                if let Some(axis) = axis {
                    let kths = normalize_kths(array.shape(), &kths, axis)?;
                    array.argpartition_many(&kths, axis)
                } else {
                    let array = array.flatten();
                    let kths = normalize_kths(array.shape(), &kths, 0)?;
                    array.argpartition_many(&kths, 0)
                }
            })
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
