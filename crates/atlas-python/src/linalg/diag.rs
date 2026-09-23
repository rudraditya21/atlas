use atlas_ndarray::{AtlasNdResult, NDArray, Numeric, checked_element_count};
use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{array, gil};

pub(crate) fn diag(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    array::require_numpy_array(py, value)?;
    let dtype: String = value.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let value = array::from_numpy(array::readonly_from_python::<$ty>(py, value)?)?;
            let result = gil::without_gil(py, move || diag_array(value))
                .map_err(|error| crate::error::linalg(py, error))?;
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
        "bool" => Err(PyTypeError::new_err("diag does not support bool dtype")),
        _ => Err(PyTypeError::new_err(format!("unsupported NumPy dtype {dtype}"))),
    }
}

fn diag_array<T: Numeric>(array: NDArray<T>) -> atlas_linalg::AtlasLinalgResult<NDArray<T>> {
    if array.ndim() == 1 {
        return diagonal_matrix(array).map_err(Into::into);
    }

    atlas_linalg::diag(&array, 0)
}

fn diagonal_matrix<T: Numeric>(array: NDArray<T>) -> AtlasNdResult<NDArray<T>> {
    let length = array.len();
    let mut values = vec![T::zero(); checked_element_count(&[length, length])?];

    for (index, value) in array.data().iter().copied().enumerate() {
        values[index * length + index] = value;
    }

    NDArray::from_shape_vec([length, length], values)
}
