use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::support::{
    arrays as array,
    dtypes::{DType, with_dtype},
    gil,
};

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
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    apply(py, lhs, rhs, out, Operation::Add)
}

pub(crate) fn subtract(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    apply(py, lhs, rhs, out, Operation::Subtract)
}

pub(crate) fn multiply(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    apply(py, lhs, rhs, out, Operation::Multiply)
}

pub(crate) fn divide(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    apply(py, lhs, rhs, out, Operation::Divide)
}

fn apply(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    out: Option<&Bound<'_, PyAny>>,
    operation: Operation,
) -> PyResult<Py<PyAny>> {
    let lhs_dtype = array::source_dtype(py, lhs)?;
    if array::is_numpy_array(py, rhs)? {
        let rhs_dtype = array::source_dtype(py, rhs)?;
        let promoted = lhs_dtype.promote_with(rhs_dtype);
        if matches!(promoted, DType::Bool) {
            return Err(PyTypeError::new_err("arithmetic does not support bool dtype"));
        }

        return with_dtype!(
            promoted,
            numeric | T | {
                let lhs = array::from_numpy_promoted::<T>(py, lhs, lhs_dtype)?;
                let rhs = array::from_numpy_promoted::<T>(py, rhs, rhs_dtype)?;
                let result = gil::without_gil(py, move || match operation {
                    Operation::Add => &lhs + &rhs,
                    Operation::Subtract => &lhs - &rhs,
                    Operation::Multiply => &lhs * &rhs,
                    Operation::Divide => &lhs / &rhs,
                })
                .map_err(|error| crate::support::errors::ndarray(py, error))?;
                array::to_numpy_output(py, result, out)
            }
        );
    }

    match lhs_dtype {
        DType::Int8 => apply_i8(py, lhs, rhs, out, operation),
        DType::Int16 => apply_i16(py, lhs, rhs, out, operation),
        DType::Int32 => apply_i32(py, lhs, rhs, out, operation),
        DType::Int64 => apply_i64(py, lhs, rhs, out, operation),
        DType::UInt8 => apply_u8(py, lhs, rhs, out, operation),
        DType::UInt16 => apply_u16(py, lhs, rhs, out, operation),
        DType::UInt32 => apply_u32(py, lhs, rhs, out, operation),
        DType::UInt64 => apply_u64(py, lhs, rhs, out, operation),
        DType::Float32 => apply_f32(py, lhs, rhs, out, operation),
        DType::Float64 => apply_f64(py, lhs, rhs, out, operation),
        DType::Bool => Err(PyTypeError::new_err("arithmetic does not support bool dtype")),
    }
}

macro_rules! impl_apply {
    ($name:ident, $ty:ty) => {
        fn $name(
            py: Python<'_>,
            lhs: &Bound<'_, PyAny>,
            rhs: &Bound<'_, PyAny>,
            out: Option<&Bound<'_, PyAny>>,
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

            let result = result.map_err(|error| crate::support::errors::ndarray(py, error))?;
            array::to_numpy_output(py, result, out)
        }
    };
}

impl_apply!(apply_i64, i64);
impl_apply!(apply_i8, i8);
impl_apply!(apply_i16, i16);
impl_apply!(apply_i32, i32);
impl_apply!(apply_u8, u8);
impl_apply!(apply_u16, u16);
impl_apply!(apply_u32, u32);
impl_apply!(apply_u64, u64);
impl_apply!(apply_f32, f32);
impl_apply!(apply_f64, f64);
