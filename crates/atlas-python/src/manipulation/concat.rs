use atlas_ndarray::{AtlasNdError, NDArray};
use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{array, gil};

pub(crate) fn concatenate(
    py: Python<'_>,
    arrays: Vec<Py<PyAny>>,
    axis: i64,
) -> PyResult<Py<PyAny>> {
    let first = arrays.first().ok_or_else(|| {
        crate::error::ndarray(
            py,
            AtlasNdError::InvalidArgument {
                op: "concatenate",
                reason: "at least one array is required",
            },
        )
    })?;
    let first = first.bind(py);
    array::require_numpy_array(py, first)?;
    let dtype: String = first.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let arrays = arrays
                .iter()
                .map(|value| {
                    array::readonly_from_python::<$ty>(py, value.bind(py))
                        .and_then(array::from_numpy)
                })
                .collect::<PyResult<Vec<_>>>()?;
            let result = gil::without_gil(py, move || {
                let views = arrays.iter().map(NDArray::view).collect::<Vec<_>>();
                NDArray::concatenate(&views, axis)
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
