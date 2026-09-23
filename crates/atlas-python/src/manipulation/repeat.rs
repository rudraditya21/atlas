use atlas_ndarray::AtlasNdError;
use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{array, gil};

pub(crate) fn repeat(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    repeats: i64,
    axis: Option<i64>,
) -> PyResult<Py<PyAny>> {
    let repeats = usize::try_from(repeats).map_err(|_| {
        crate::error::ndarray(
            py,
            AtlasNdError::InvalidArgument { op: "repeat", reason: "repeats must be nonnegative" },
        )
    })?;
    array::require_numpy_array(py, value)?;
    let dtype: String = value.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let array = array::from_numpy(array::readonly_from_python::<$ty>(py, value)?)?;
            let result = gil::without_gil(py, move || match axis {
                Some(axis) => array.repeat(repeats, axis),
                None => array.flatten().repeat(repeats, 0),
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
