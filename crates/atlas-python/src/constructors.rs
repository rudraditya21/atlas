use atlas_ndarray::NDArray;
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::{array, python_dtype::DType};

pub(crate) fn asarray(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let dtype = dtype.map(str::to_owned).or_else(|| {
        value
            .getattr("dtype")
            .ok()
            .and_then(|dtype| dtype.getattr("name").ok())
            .and_then(|name| name.extract::<String>().ok())
    });

    match DType::parse(dtype.as_deref())? {
        DType::Bool => asarray_typed::<bool>(py, value),
        DType::Int8 => asarray_typed::<i8>(py, value),
        DType::Int16 => asarray_typed::<i16>(py, value),
        DType::Int32 => asarray_typed::<i32>(py, value),
        DType::Int64 => asarray_typed::<i64>(py, value),
        DType::UInt8 => asarray_typed::<u8>(py, value),
        DType::UInt16 => asarray_typed::<u16>(py, value),
        DType::UInt32 => asarray_typed::<u32>(py, value),
        DType::UInt64 => asarray_typed::<u64>(py, value),
        DType::Float32 => asarray_typed::<f32>(py, value),
        DType::Float64 => asarray_typed::<f64>(py, value),
    }
}

pub(crate) fn zeros(py: Python<'_>, shape: Vec<usize>, dtype: Option<&str>) -> PyResult<Py<PyAny>> {
    match DType::parse(dtype)? {
        DType::Bool => output(py, NDArray::full(shape, false)),
        DType::Int8 => output(py, NDArray::<i8>::zeros(shape)),
        DType::Int16 => output(py, NDArray::<i16>::zeros(shape)),
        DType::Int32 => output(py, NDArray::<i32>::zeros(shape)),
        DType::Int64 => output(py, NDArray::<i64>::zeros(shape)),
        DType::UInt8 => output(py, NDArray::<u8>::zeros(shape)),
        DType::UInt16 => output(py, NDArray::<u16>::zeros(shape)),
        DType::UInt32 => output(py, NDArray::<u32>::zeros(shape)),
        DType::UInt64 => output(py, NDArray::<u64>::zeros(shape)),
        DType::Float32 => output(py, NDArray::<f32>::zeros(shape)),
        DType::Float64 => output(py, NDArray::<f64>::zeros(shape)),
    }
}

pub(crate) fn ones(py: Python<'_>, shape: Vec<usize>, dtype: Option<&str>) -> PyResult<Py<PyAny>> {
    match DType::parse(dtype)? {
        DType::Bool => output(py, NDArray::full(shape, true)),
        DType::Int8 => output(py, NDArray::<i8>::ones(shape)),
        DType::Int16 => output(py, NDArray::<i16>::ones(shape)),
        DType::Int32 => output(py, NDArray::<i32>::ones(shape)),
        DType::Int64 => output(py, NDArray::<i64>::ones(shape)),
        DType::UInt8 => output(py, NDArray::<u8>::ones(shape)),
        DType::UInt16 => output(py, NDArray::<u16>::ones(shape)),
        DType::UInt32 => output(py, NDArray::<u32>::ones(shape)),
        DType::UInt64 => output(py, NDArray::<u64>::ones(shape)),
        DType::Float32 => output(py, NDArray::<f32>::ones(shape)),
        DType::Float64 => output(py, NDArray::<f64>::ones(shape)),
    }
}

pub(crate) fn full(
    py: Python<'_>,
    shape: Vec<usize>,
    value: &Bound<'_, PyAny>,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    match DType::parse(dtype)? {
        DType::Bool => output(py, NDArray::full(shape, value.extract::<bool>()?)),
        DType::Int8 => output(py, NDArray::full(shape, value.extract::<i8>()?)),
        DType::Int16 => output(py, NDArray::full(shape, value.extract::<i16>()?)),
        DType::Int32 => output(py, NDArray::full(shape, value.extract::<i32>()?)),
        DType::Int64 => output(py, NDArray::full(shape, value.extract::<i64>()?)),
        DType::UInt8 => output(py, NDArray::full(shape, value.extract::<u8>()?)),
        DType::UInt16 => output(py, NDArray::full(shape, value.extract::<u16>()?)),
        DType::UInt32 => output(py, NDArray::full(shape, value.extract::<u32>()?)),
        DType::UInt64 => output(py, NDArray::full(shape, value.extract::<u64>()?)),
        DType::Float32 => output(py, NDArray::full(shape, value.extract::<f32>()?)),
        DType::Float64 => output(py, NDArray::full(shape, value.extract::<f64>()?)),
    }
}

pub(crate) fn arange(
    py: Python<'_>,
    start: f64,
    stop: Option<f64>,
    step: Option<f64>,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let (start, stop) = stop.map_or((0.0, start), |stop| (start, stop));
    let step = step.unwrap_or(1.0);

    macro_rules! integer_arange {
        ($ty:ty, $min:expr, $max:expr) => {{
            let start = checked_integer_arange_value(start, "start", $min, $max)? as $ty;
            let stop = checked_integer_arange_value(stop, "stop", $min, $max)? as $ty;
            let step = checked_integer_arange_value(step, "step", $min, $max)? as $ty;
            output(py, NDArray::arange(start, stop, step))
        }};
    }

    match DType::parse(dtype)? {
        DType::Bool => Err(PyValueError::new_err("arange does not support bool dtype")),
        DType::Int8 => integer_arange!(i8, i8::MIN as f64, i8::MAX as f64 + 1.0),
        DType::Int16 => integer_arange!(i16, i16::MIN as f64, i16::MAX as f64 + 1.0),
        DType::Int32 => integer_arange!(i32, i32::MIN as f64, i32::MAX as f64 + 1.0),
        DType::Int64 => integer_arange!(i64, i64::MIN as f64, i64::MAX as f64),
        DType::UInt8 => integer_arange!(u8, 0.0, u8::MAX as f64 + 1.0),
        DType::UInt16 => integer_arange!(u16, 0.0, u16::MAX as f64 + 1.0),
        DType::UInt32 => integer_arange!(u32, 0.0, u32::MAX as f64 + 1.0),
        DType::UInt64 => integer_arange!(u64, 0.0, u64::MAX as f64),
        DType::Float32 => output(py, NDArray::arange(start as f32, stop as f32, step as f32)),
        DType::Float64 => output(py, NDArray::arange(start, stop, step)),
    }
}

fn checked_integer_arange_value(
    value: f64,
    name: &str,
    minimum: f64,
    maximum_exclusive: f64,
) -> PyResult<f64> {
    if !value.is_finite() || value.fract() != 0.0 || value < minimum || value >= maximum_exclusive {
        return Err(PyValueError::new_err(format!(
            "integer arange {name} must be a finite integral value within the target dtype range"
        )));
    }

    Ok(value)
}

fn asarray_typed<T>(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>>
where
    T: atlas_ndarray::ArrayElement + numpy::Element,
{
    output_owned(py, array::from_numpy(array::readonly_from_python::<T>(py, value)?)?)
}

fn output<T>(py: Python<'_>, array: atlas_ndarray::AtlasNdResult<NDArray<T>>) -> PyResult<Py<PyAny>>
where
    T: atlas_ndarray::ArrayElement + numpy::Element,
{
    let array = array.map_err(|error| crate::error::ndarray(py, error))?;

    output_owned(py, array)
}

fn output_owned<T>(py: Python<'_>, array: NDArray<T>) -> PyResult<Py<PyAny>>
where
    T: atlas_ndarray::ArrayElement + numpy::Element,
{
    Ok(array::to_numpy_owned(py, array)?.into_any().unbind())
}
