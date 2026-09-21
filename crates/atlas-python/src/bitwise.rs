use atlas_ndarray::BitwiseElement;
use numpy::Element;
use pyo3::{FromPyObject, exceptions::PyTypeError, prelude::*};

use crate::{array, gil};

#[derive(Clone, Copy)]
enum BinaryOperation {
    And,
    Or,
    Xor,
}

pub(crate) fn bitwise_and(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    binary(py, lhs, rhs, BinaryOperation::And)
}

pub(crate) fn bitwise_or(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    binary(py, lhs, rhs, BinaryOperation::Or)
}

pub(crate) fn bitwise_xor(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    binary(py, lhs, rhs, BinaryOperation::Xor)
}

pub(crate) fn bitwise_not(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    array::require_numpy_array(py, value)?;
    let dtype: String = value.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let value = array::from_numpy(array::readonly_from_python::<$ty>(py, value)?)?;
            output_owned(py, gil::without_gil(py, move || value.bitnot()))
        }};
    }

    match dtype.as_str() {
        "bool" => {
            let value = array::from_numpy(array::readonly_from_python::<bool>(py, value)?)?;
            output_owned(py, gil::without_gil(py, move || value.logical_not()))
        }
        "int8" => apply!(i8),
        "int16" => apply!(i16),
        "int32" => apply!(i32),
        "int64" => apply!(i64),
        "uint8" => apply!(u8),
        "uint16" => apply!(u16),
        "uint32" => apply!(u32),
        "uint64" => apply!(u64),
        _ => Err(PyTypeError::new_err("bitwise operations require an integer or bool NumPy array")),
    }
}

fn binary(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    operation: BinaryOperation,
) -> PyResult<Py<PyAny>> {
    array::require_numpy_array(py, lhs)?;
    let dtype: String = lhs.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{ integer_binary::<$ty>(py, lhs, rhs, operation) }};
    }

    match dtype.as_str() {
        "bool" => boolean_binary(py, lhs, rhs, operation),
        "int8" => apply!(i8),
        "int16" => apply!(i16),
        "int32" => apply!(i32),
        "int64" => apply!(i64),
        "uint8" => apply!(u8),
        "uint16" => apply!(u16),
        "uint32" => apply!(u32),
        "uint64" => apply!(u64),
        _ => Err(PyTypeError::new_err("bitwise operations require an integer or bool NumPy array")),
    }
}

fn integer_binary<T>(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    operation: BinaryOperation,
) -> PyResult<Py<PyAny>>
where
    T: BitwiseElement + Element,
    for<'a, 'py> T: FromPyObject<'a, 'py>,
{
    let lhs = array::from_numpy(array::readonly_from_python::<T>(py, lhs)?)?;
    if array::is_numpy_array(py, rhs)? {
        let rhs = array::from_numpy(array::readonly_from_python::<T>(py, rhs)?)?;
        let result = gil::without_gil(py, move || match operation {
            BinaryOperation::And => lhs.bitand(&rhs),
            BinaryOperation::Or => lhs.bitor(&rhs),
            BinaryOperation::Xor => lhs.bitxor(&rhs),
        });
        output(py, result)
    } else {
        let rhs = rhs.extract::<T>().map_err(Into::<PyErr>::into)?;
        let result = gil::without_gil(py, move || match operation {
            BinaryOperation::And => lhs.bitand(rhs),
            BinaryOperation::Or => lhs.bitor(rhs),
            BinaryOperation::Xor => lhs.bitxor(rhs),
        });
        output_owned(py, result)
    }
}

fn boolean_binary(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    operation: BinaryOperation,
) -> PyResult<Py<PyAny>> {
    let lhs = array::from_numpy(array::readonly_from_python::<bool>(py, lhs)?)?;
    if array::is_numpy_array(py, rhs)? {
        let rhs = array::from_numpy(array::readonly_from_python::<bool>(py, rhs)?)?;
        let result = gil::without_gil(py, move || match operation {
            BinaryOperation::And => lhs.logical_and(&rhs),
            BinaryOperation::Or => lhs.logical_or(&rhs),
            BinaryOperation::Xor => lhs.logical_xor(&rhs),
        });
        output(py, result)
    } else {
        let rhs = rhs.extract::<bool>()?;
        let result = gil::without_gil(py, move || match operation {
            BinaryOperation::And => lhs.logical_and(rhs),
            BinaryOperation::Or => lhs.logical_or(rhs),
            BinaryOperation::Xor => lhs.logical_xor(rhs),
        });
        output_owned(py, result)
    }
}

fn output<T>(
    py: Python<'_>,
    result: atlas_ndarray::AtlasNdResult<atlas_ndarray::NDArray<T>>,
) -> PyResult<Py<PyAny>>
where
    T: atlas_ndarray::ArrayElement + Element,
{
    output_owned(py, result.map_err(|error| crate::error::ndarray(py, error))?)
}

fn output_owned<T>(py: Python<'_>, array: atlas_ndarray::NDArray<T>) -> PyResult<Py<PyAny>>
where
    T: atlas_ndarray::ArrayElement + Element,
{
    Ok(array::to_numpy_owned(py, array)?.into_any().unbind())
}
