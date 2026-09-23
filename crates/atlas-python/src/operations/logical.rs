use pyo3::{IntoPyObjectExt, exceptions::PyTypeError, prelude::*, types::PyTuple};

use crate::{array, gil, python_dtype::with_dtype};

#[derive(Clone, Copy)]
enum Comparison {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

macro_rules! with_array {
    ($py:expr, $value:expr, |$array:ident| $body:expr) => {{
        let dtype = array::source_dtype($py, $value)?;
        with_dtype!(
            dtype,
            all | T | {
                let $array = array::from_numpy(array::readonly_from_python::<T>($py, $value)?)?;
                $body
            }
        )
    }};
}

pub(crate) fn equal(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    compare(py, lhs, rhs, Comparison::Equal)
}

pub(crate) fn not_equal(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    compare(py, lhs, rhs, Comparison::NotEqual)
}

pub(crate) fn less(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    compare(py, lhs, rhs, Comparison::Less)
}

pub(crate) fn less_equal(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    compare(py, lhs, rhs, Comparison::LessEqual)
}

pub(crate) fn greater(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    compare(py, lhs, rhs, Comparison::Greater)
}

pub(crate) fn greater_equal(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    compare(py, lhs, rhs, Comparison::GreaterEqual)
}

pub(crate) fn count_true(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<usize> {
    let value = boolean_array(py, value)?;
    Ok(gil::without_gil(py, move || value.count_true()))
}

pub(crate) fn all(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    axis: Option<i64>,
) -> PyResult<Py<PyAny>> {
    let value = boolean_array(py, value)?;
    match axis {
        Some(axis) => output(py, gil::without_gil(py, move || value.all_axis(axis))),
        None => gil::without_gil(py, move || value.all()).into_py_any(py),
    }
}

pub(crate) fn any(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    axis: Option<i64>,
) -> PyResult<Py<PyAny>> {
    let value = boolean_array(py, value)?;
    match axis {
        Some(axis) => output(py, gil::without_gil(py, move || value.any_axis(axis))),
        None => gil::without_gil(py, move || value.any()).into_py_any(py),
    }
}

pub(crate) fn all_axis(py: Python<'_>, value: &Bound<'_, PyAny>, axis: i64) -> PyResult<Py<PyAny>> {
    all(py, value, Some(axis))
}

pub(crate) fn any_axis(py: Python<'_>, value: &Bound<'_, PyAny>, axis: i64) -> PyResult<Py<PyAny>> {
    any(py, value, Some(axis))
}

pub(crate) fn select(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    mask: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    with_array!(py, value, |array| {
        let mask = boolean_array(py, mask)?;
        output(py, gil::without_gil(py, move || array.select(&mask)))
    })
}

pub(crate) fn nonzero(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    with_array!(py, value, |array| {
        let indices = gil::without_gil(py, move || array.nonzero_indices())
            .map_err(|error| crate::error::ndarray(py, error))?;
        let (data, shape) = indices.into_raw_parts();
        let matches = shape[0];
        let dimensions = shape[1];
        if dimensions == 0 {
            return Ok(PyTuple::empty(py).into_any().unbind());
        }
        let mut per_axis = vec![Vec::with_capacity(matches); dimensions];
        for coordinate in data.chunks(dimensions) {
            for (axis, &index) in coordinate.iter().enumerate() {
                per_axis[axis].push(i64::try_from(index).expect("NumPy dimensions fit i64"));
            }
        }
        let indices = per_axis
            .into_iter()
            .map(|indices| {
                array::to_numpy_owned(
                    py,
                    atlas_ndarray::NDArray::from_shape_vec([matches], indices)
                        .expect("nonzero preserves ndarray invariants"),
                )
                .map(|array| array.into_any().unbind())
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, indices)?.into_any().unbind())
    })
}

pub(crate) fn argwhere(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    with_array!(py, value, |array| {
        let indices = gil::without_gil(py, move || array.argwhere());
        let (data, shape) = indices.into_raw_parts();
        let data = data
            .into_iter()
            .map(|index| i64::try_from(index).expect("NumPy dimensions fit i64"))
            .collect();
        output_owned(
            py,
            atlas_ndarray::NDArray::from_shape_vec(shape, data)
                .expect("argwhere preserves ndarray invariants"),
        )
    })
}

pub(crate) fn masked_fill(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    mask: &Bound<'_, PyAny>,
    fill: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    with_array!(py, value, |array| {
        let mask = boolean_array(py, mask)?;
        let fill = fill.extract()?;
        let array = gil::without_gil(py, move || {
            let mut array = array;
            array.masked_fill(&mask, fill)?;
            Ok::<_, atlas_ndarray::AtlasNdError>(array)
        })
        .map_err(|error| crate::error::ndarray(py, error))?;
        output_owned(py, array)
    })
}

fn compare(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    comparison: Comparison,
) -> PyResult<Py<PyAny>> {
    array::require_numpy_array(py, lhs)?;
    let dtype: String = lhs.getattr("dtype")?.getattr("name")?.extract()?;

    match dtype.as_str() {
        "bool" => compare_bool(py, lhs, rhs, comparison),
        "int8" => compare_i8(py, lhs, rhs, comparison),
        "int16" => compare_i16(py, lhs, rhs, comparison),
        "int32" => compare_i32(py, lhs, rhs, comparison),
        "int64" => compare_i64(py, lhs, rhs, comparison),
        "uint8" => compare_u8(py, lhs, rhs, comparison),
        "uint16" => compare_u16(py, lhs, rhs, comparison),
        "uint32" => compare_u32(py, lhs, rhs, comparison),
        "uint64" => compare_u64(py, lhs, rhs, comparison),
        "float32" => compare_f32(py, lhs, rhs, comparison),
        "float64" => compare_f64(py, lhs, rhs, comparison),
        _ => Err(PyTypeError::new_err(format!("unsupported NumPy dtype {dtype}"))),
    }
}

fn boolean_array(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<atlas_ndarray::NDArray<bool>> {
    array::from_numpy(array::readonly_from_python::<bool>(py, value)?)
}

macro_rules! impl_compare {
    ($name:ident, $ty:ty) => {
        fn $name(
            py: Python<'_>,
            lhs: &Bound<'_, PyAny>,
            rhs: &Bound<'_, PyAny>,
            comparison: Comparison,
        ) -> PyResult<Py<PyAny>> {
            let lhs = array::from_numpy(array::readonly_from_python::<$ty>(py, lhs)?)?;
            let result = if array::is_numpy_array(py, rhs)? {
                let rhs = array::from_numpy(array::readonly_from_python::<$ty>(py, rhs)?)?;
                gil::without_gil(py, move || match comparison {
                    Comparison::Equal => lhs.eq(&rhs),
                    Comparison::NotEqual => lhs.ne(&rhs),
                    Comparison::Less => lhs.lt(&rhs),
                    Comparison::LessEqual => lhs.le(&rhs),
                    Comparison::Greater => lhs.gt(&rhs),
                    Comparison::GreaterEqual => lhs.ge(&rhs),
                })
            } else {
                let rhs = rhs.extract::<$ty>()?;
                gil::without_gil(py, move || {
                    Ok(match comparison {
                        Comparison::Equal => lhs.eq(rhs),
                        Comparison::NotEqual => lhs.ne(rhs),
                        Comparison::Less => lhs.lt(rhs),
                        Comparison::LessEqual => lhs.le(rhs),
                        Comparison::Greater => lhs.gt(rhs),
                        Comparison::GreaterEqual => lhs.ge(rhs),
                    })
                })
            };

            output(py, result)
        }
    };
}

impl_compare!(compare_bool, bool);
impl_compare!(compare_i8, i8);
impl_compare!(compare_i16, i16);
impl_compare!(compare_i32, i32);
impl_compare!(compare_i64, i64);
impl_compare!(compare_u8, u8);
impl_compare!(compare_u16, u16);
impl_compare!(compare_u32, u32);
impl_compare!(compare_u64, u64);
impl_compare!(compare_f32, f32);
impl_compare!(compare_f64, f64);

fn output<T>(
    py: Python<'_>,
    array: atlas_ndarray::AtlasNdResult<atlas_ndarray::NDArray<T>>,
) -> PyResult<Py<PyAny>>
where
    T: atlas_ndarray::ArrayElement + numpy::Element,
{
    output_owned(py, array.map_err(|error| crate::error::ndarray(py, error))?)
}

fn output_owned<T>(py: Python<'_>, array: atlas_ndarray::NDArray<T>) -> PyResult<Py<PyAny>>
where
    T: atlas_ndarray::ArrayElement + numpy::Element,
{
    Ok(array::to_numpy_owned(py, array)?.into_any().unbind())
}
