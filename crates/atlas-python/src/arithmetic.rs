use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};

use crate::{array, gil};

#[derive(Clone, Copy)]
enum Operation {
    Add,
    Subtract,
    Multiply,
    Divide,
}

pub(crate) fn add(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    apply(py, lhs, rhs, Operation::Add)
}

pub(crate) fn subtract(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    apply(py, lhs, rhs, Operation::Subtract)
}

pub(crate) fn multiply(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    apply(py, lhs, rhs, Operation::Multiply)
}

pub(crate) fn divide(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    apply(py, lhs, rhs, Operation::Divide)
}

fn apply(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    operation: Operation,
) -> PyResult<Py<PyAny>> {
    array::require_numpy_array(py, lhs)?;
    let dtype: String = lhs.getattr("dtype")?.getattr("name")?.extract()?;

    match dtype.as_str() {
        "int64" => apply_i64(py, lhs, rhs, operation),
        "float32" => apply_f32(py, lhs, rhs, operation),
        "float64" => apply_f64(py, lhs, rhs, operation),
        "bool" => Err(PyTypeError::new_err("arithmetic does not support bool dtype")),
        _ => Err(PyTypeError::new_err(format!("unsupported NumPy dtype {dtype}"))),
    }
}

macro_rules! impl_apply {
    ($name:ident, $ty:ty) => {
        fn $name(
            py: Python<'_>,
            lhs: &Bound<'_, PyAny>,
            rhs: &Bound<'_, PyAny>,
            operation: Operation,
        ) -> PyResult<Py<PyAny>> {
            let lhs = array::from_numpy(array::readonly_from_python::<$ty>(py, lhs)?)?;
            let result = match operation {
                Operation::Add => {
                    if array::is_numpy_array(py, rhs)? {
                        let rhs = array::from_numpy(array::readonly_from_python::<$ty>(py, rhs)?)?;
                        gil::without_gil(py, move || &lhs + &rhs)
                    } else {
                        let rhs = rhs.extract::<$ty>()?;
                        gil::without_gil(py, move || Ok(&lhs + rhs))
                    }
                }
                Operation::Subtract => {
                    if array::is_numpy_array(py, rhs)? {
                        let rhs = array::from_numpy(array::readonly_from_python::<$ty>(py, rhs)?)?;
                        gil::without_gil(py, move || &lhs - &rhs)
                    } else {
                        let rhs = rhs.extract::<$ty>()?;
                        gil::without_gil(py, move || Ok(&lhs - rhs))
                    }
                }
                Operation::Multiply => {
                    if array::is_numpy_array(py, rhs)? {
                        let rhs = array::from_numpy(array::readonly_from_python::<$ty>(py, rhs)?)?;
                        gil::without_gil(py, move || &lhs * &rhs)
                    } else {
                        let rhs = rhs.extract::<$ty>()?;
                        gil::without_gil(py, move || Ok(&lhs * rhs))
                    }
                }
                Operation::Divide => {
                    if array::is_numpy_array(py, rhs)? {
                        let rhs = array::from_numpy(array::readonly_from_python::<$ty>(py, rhs)?)?;
                        gil::without_gil(py, move || &lhs / &rhs)
                    } else {
                        let rhs = rhs.extract::<$ty>()?;
                        gil::without_gil(py, move || &lhs / rhs)
                    }
                }
            };

            let result = result.map_err(|error| PyValueError::new_err(error.to_string()))?;
            Ok(array::to_numpy_owned(py, result)?.into_any().unbind())
        }
    };
}

impl_apply!(apply_i64, i64);
impl_apply!(apply_f32, f32);
impl_apply!(apply_f64, f64);
