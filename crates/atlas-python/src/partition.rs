use atlas_ndarray::AtlasNdError;
use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{array, gil};
pub(crate) fn partition(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    kth: i64,
    axis: Option<i64>,
) -> PyResult<Py<PyAny>> {
    array::require_numpy_array(py, value)?;
    let dtype: String = value.getattr("dtype")?.getattr("name")?.extract()?;
    macro_rules! apply {
        ($ty:ty) => {{
            let array = array::from_numpy(array::readonly_from_python::<$ty>(py, value)?)?;
            let result = gil::without_gil(py, move || {
                if let Some(axis) = axis {
                    let kth = normalize_kth(array.shape(), kth, axis)?;
                    array.partition(kth, axis)
                } else {
                    let array = array.flatten();
                    let kth = normalize_kth(array.shape(), kth, 0)?;
                    array.partition(kth, 0)
                }
            })
            .map_err(|e| crate::error::ndarray(py, e))?;
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
pub(crate) fn normalize_kth(shape: &[usize], kth: i64, axis: i64) -> Result<usize, AtlasNdError> {
    let n = shape.len();
    let a = if axis < 0 { axis + n as i64 } else { axis };
    if a < 0 || a as usize >= n {
        return Err(AtlasNdError::InvalidAxis { axis: a, ndim: n });
    }
    let len = shape[a as usize];
    let k = if kth < 0 { kth + len as i64 } else { kth };
    if k < 0 || k as usize >= len {
        return Err(AtlasNdError::IndexOutOfBounds { axis: a as usize, index: k, dim: len });
    }
    Ok(k as usize)
}
