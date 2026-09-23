use std::cmp::Ordering;

use atlas_ndarray::{AtlasNdError, NDArray, SortElement};
use pyo3::{
    IntoPyObjectExt,
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};

use crate::{array, gil};

#[derive(Clone, Copy)]
enum Side {
    Left,
    Right,
}

pub(crate) fn searchsorted(
    py: Python<'_>,
    sorted: &Bound<'_, PyAny>,
    values: &Bound<'_, PyAny>,
    side: &str,
) -> PyResult<Py<PyAny>> {
    let side = match side {
        "left" => Side::Left,
        "right" => Side::Right,
        _ => return Err(PyValueError::new_err("side must be 'left' or 'right'")),
    };
    array::require_numpy_array(py, sorted)?;
    let dtype: String = sorted.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let sorted = array::from_numpy(array::readonly_from_python::<$ty>(py, sorted)?)?;
            let (values, shape) = if array::is_numpy_array(py, values)? {
                let values = array::from_numpy(array::readonly_from_python::<$ty>(py, values)?)?;
                (values.data().to_vec(), Some(values.shape().to_vec()))
            } else {
                (vec![values.extract::<$ty>()?], None)
            };
            let indices = gil::without_gil(py, move || search_indices(&sorted, &values, side))
                .map_err(|error| crate::error::ndarray(py, error))?;
            match shape {
                Some(shape) => Ok(array::to_numpy_owned(
                    py,
                    NDArray::from_shape_vec(shape, indices)
                        .expect("searchsorted preserves ndarray invariants"),
                )?
                .into_any()
                .unbind()),
                None => indices[0].into_py_any(py),
            }
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

fn search_indices<T: SortElement>(
    sorted: &NDArray<T>,
    values: &[T],
    side: Side,
) -> Result<Vec<i64>, AtlasNdError> {
    if sorted.ndim() != 1 {
        return Err(AtlasNdError::DimensionMismatch { expected: 1, actual: sorted.ndim() });
    }
    values
        .iter()
        .map(|value| {
            let mut low = 0;
            let mut high = sorted.data().len();
            while low < high {
                let middle = low + (high - low) / 2;
                let ordering = sorted.data()[middle].sort_compare(value);
                if ordering == Ordering::Less
                    || (matches!(side, Side::Right) && ordering == Ordering::Equal)
                {
                    low = middle + 1;
                } else {
                    high = middle;
                }
            }
            Ok(i64::try_from(low).expect("ndarray indices fit i64"))
        })
        .collect()
}
