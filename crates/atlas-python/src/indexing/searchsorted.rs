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
    sorter: Option<&Bound<'_, PyAny>>,
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
            let sorter = sorter_indices(py, sorter, sorted.size())?;
            let (values, shape) = if array::is_numpy_array(py, values)? {
                let values = array::from_numpy(array::readonly_from_python::<$ty>(py, values)?)?;
                (values.data().to_vec(), Some(values.shape().to_vec()))
            } else {
                (vec![values.extract::<$ty>()?], None)
            };
            let indices = gil::without_gil(py, move || {
                search_indices(&sorted, &values, side, sorter.as_deref())
            })
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

fn sorter_indices(
    py: Python<'_>,
    sorter: Option<&Bound<'_, PyAny>>,
    length: usize,
) -> PyResult<Option<Vec<usize>>> {
    let Some(sorter) = sorter else {
        return Ok(None);
    };
    array::require_numpy_array(py, sorter)?;
    let dimensions: usize = sorter.getattr("ndim")?.extract()?;
    if dimensions != 1 {
        return Err(PyValueError::new_err("sorter must be one-dimensional"));
    }

    macro_rules! indices {
        ($ty:ty) => {{
            let sorter = array::from_numpy(array::readonly_from_python::<$ty>(py, sorter)?)?;
            sorter
                .data()
                .iter()
                .copied()
                .map(|index| {
                    usize::try_from(index)
                        .map_err(|_| PyValueError::new_err("sorter indices must be in bounds"))
                })
                .collect::<PyResult<Vec<_>>>()
        }};
    }

    let dtype: String = sorter.getattr("dtype")?.getattr("name")?.extract()?;
    let indices = match dtype.as_str() {
        "int8" => indices!(i8),
        "int16" => indices!(i16),
        "int32" => indices!(i32),
        "int64" => indices!(i64),
        "uint8" => indices!(u8),
        "uint16" => indices!(u16),
        "uint32" => indices!(u32),
        "uint64" => indices!(u64),
        _ => return Err(PyTypeError::new_err("sorter must have an integer dtype")),
    }?;
    if indices.len() != length {
        return Err(PyValueError::new_err("sorter must have the same length as sorted"));
    }

    let mut seen = vec![false; length];
    for &index in &indices {
        if index >= length {
            return Err(PyValueError::new_err("sorter indices must be in bounds"));
        }
        if std::mem::replace(&mut seen[index], true) {
            return Err(PyValueError::new_err("sorter must be a permutation"));
        }
    }
    Ok(Some(indices))
}

fn search_indices<T: SortElement>(
    sorted: &NDArray<T>,
    values: &[T],
    side: Side,
    sorter: Option<&[usize]>,
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
                let index = sorter.map_or(middle, |sorter| sorter[middle]);
                let ordering = sorted.data()[index].sort_compare(value);
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
