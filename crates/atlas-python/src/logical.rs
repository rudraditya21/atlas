use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};

use crate::array;

#[derive(Clone, Copy)]
enum Comparison {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

macro_rules! with_dtype {
    ($py:expr, $value:expr, |$array:ident| $body:expr) => {{
        array::require_numpy_array($py, $value)?;
        let dtype: String = $value.getattr("dtype")?.getattr("name")?.extract()?;
        match dtype.as_str() {
            "bool" => {
                let $array = array::from_numpy(array::readonly_from_python::<bool>($py, $value)?)?;
                $body
            }
            "int64" => {
                let $array = array::from_numpy(array::readonly_from_python::<i64>($py, $value)?)?;
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
    Ok(boolean_array(py, value)?.count_true())
}

pub(crate) fn select(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    mask: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    with_dtype!(py, value, |array| {
        let mask = boolean_array(py, mask)?;
        output(py, array.select(&mask))
    })
}

pub(crate) fn nonzero(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    with_dtype!(py, value, |array| output(py, array.nonzero_indices()))
}

pub(crate) fn masked_fill(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    mask: &Bound<'_, PyAny>,
    fill: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    with_dtype!(py, value, |array| {
        let mut array = array;
        let mask = boolean_array(py, mask)?;
        array
            .masked_fill(&mask, fill.extract()?)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
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
        "int64" => compare_i64(py, lhs, rhs, comparison),
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
                match comparison {
                    Comparison::Equal => lhs.eq(&rhs),
                    Comparison::NotEqual => lhs.ne(&rhs),
                    Comparison::Less => lhs.lt(&rhs),
                    Comparison::LessEqual => lhs.le(&rhs),
                    Comparison::Greater => lhs.gt(&rhs),
                    Comparison::GreaterEqual => lhs.ge(&rhs),
                }
            } else {
                let rhs = rhs.extract::<$ty>()?;
                Ok(match comparison {
                    Comparison::Equal => lhs.eq(rhs),
                    Comparison::NotEqual => lhs.ne(rhs),
                    Comparison::Less => lhs.lt(rhs),
                    Comparison::LessEqual => lhs.le(rhs),
                    Comparison::Greater => lhs.gt(rhs),
                    Comparison::GreaterEqual => lhs.ge(rhs),
                })
            };

            output(py, result)
        }
    };
}

impl_compare!(compare_bool, bool);
impl_compare!(compare_i64, i64);
impl_compare!(compare_f32, f32);
impl_compare!(compare_f64, f64);

fn output<T>(
    py: Python<'_>,
    array: atlas_ndarray::AtlasNdResult<atlas_ndarray::NDArray<T>>,
) -> PyResult<Py<PyAny>>
where
    T: atlas_ndarray::ArrayElement + numpy::Element,
{
    output_owned(py, array.map_err(|error| PyValueError::new_err(error.to_string()))?)
}

fn output_owned<T>(py: Python<'_>, array: atlas_ndarray::NDArray<T>) -> PyResult<Py<PyAny>>
where
    T: atlas_ndarray::ArrayElement + numpy::Element,
{
    Ok(array::to_numpy(py, &array)?.into_any().unbind())
}
