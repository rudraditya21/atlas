use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{
    array, gil,
    python_dtype::{DType, with_dtype},
};

macro_rules! apply_unary {
    ($py:expr, $value:expr, $dtype:expr, $category:ident, $operation:ident) => {{
        with_dtype!(
            $dtype,
            $category | T | {
                let array = array::from_numpy(array::readonly_from_python::<T>($py, $value)?)?;
                output($py, gil::without_gil($py, move || array.$operation()))
            }
        )
    }};
}

pub(crate) fn neg(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let dtype = array::source_dtype(py, value)?;
    if matches!(dtype, DType::Bool) {
        return Err(PyTypeError::new_err("neg does not support bool dtype"));
    }
    if dtype.is_unsigned() {
        return Err(PyTypeError::new_err(format!("unsupported NumPy dtype {}", dtype.name())));
    }

    apply_unary!(py, value, dtype, signed, neg)
}

pub(crate) fn abs(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let dtype = array::source_dtype(py, value)?;
    if matches!(dtype, DType::Bool) {
        return Err(PyTypeError::new_err("abs does not support bool dtype"));
    }

    apply_unary!(py, value, dtype, numeric, abs)
}

pub(crate) fn sign(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let dtype = array::source_dtype(py, value)?;
    if matches!(dtype, DType::Bool) {
        return Err(PyTypeError::new_err("sign does not support bool dtype"));
    }

    apply_unary!(py, value, dtype, numeric, sign)
}

pub(crate) fn round(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    decimals: i64,
) -> PyResult<Py<PyAny>> {
    let dtype = array::source_dtype(py, value)?;
    if matches!(dtype, DType::Bool) {
        return Err(PyTypeError::new_err("round does not support bool dtype"));
    }

    match dtype {
        DType::Float32 => round_float32(py, value, decimals),
        DType::Float64 => round_float64(py, value, decimals),
        _ => apply_unary!(py, value, dtype, numeric, round),
    }
}

pub(crate) fn isnan(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let dtype = array::source_dtype(py, value)?;

    apply_unary!(py, value, dtype, all, isnan)
}

pub(crate) fn isinf(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let dtype = array::source_dtype(py, value)?;

    apply_unary!(py, value, dtype, all, isinf)
}

pub(crate) fn isfinite(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let dtype = array::source_dtype(py, value)?;

    apply_unary!(py, value, dtype, all, isfinite)
}

fn output<T>(py: Python<'_>, array: atlas_ndarray::NDArray<T>) -> PyResult<Py<PyAny>>
where
    T: atlas_ndarray::ArrayElement + numpy::Element,
{
    Ok(array::to_numpy_owned(py, array)?.into_any().unbind())
}

fn round_float32(py: Python<'_>, value: &Bound<'_, PyAny>, decimals: i64) -> PyResult<Py<PyAny>> {
    let array = array::from_numpy(array::readonly_from_python::<f32>(py, value)?)?;
    output(
        py,
        gil::without_gil(py, move || array.map(|value| round_with_decimals(value, decimals))),
    )
}

fn round_float64(py: Python<'_>, value: &Bound<'_, PyAny>, decimals: i64) -> PyResult<Py<PyAny>> {
    let array = array::from_numpy(array::readonly_from_python::<f64>(py, value)?)?;
    output(
        py,
        gil::without_gil(py, move || array.map(|value| round_with_decimals(value, decimals))),
    )
}

trait DecimalRound: Sized {
    fn round_with_decimals(self, decimals: i64) -> Self;
}

impl DecimalRound for f32 {
    fn round_with_decimals(self, decimals: i64) -> Self {
        let scale = 10_f32.powi(decimals.clamp(-38, 38) as i32);
        (self * scale).round() / scale
    }
}

impl DecimalRound for f64 {
    fn round_with_decimals(self, decimals: i64) -> Self {
        let scale = 10_f64.powi(decimals.clamp(-308, 308) as i32);
        (self * scale).round() / scale
    }
}

fn round_with_decimals<T: DecimalRound>(value: T, decimals: i64) -> T {
    value.round_with_decimals(decimals)
}
