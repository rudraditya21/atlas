use atlas_ndarray::NDArray;
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::array;

#[derive(Clone, Copy)]
enum DType {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
}

impl DType {
    fn parse(dtype: Option<&str>) -> PyResult<Self> {
        match dtype.unwrap_or("float64") {
            "bool" => Ok(Self::Bool),
            "int8" => Ok(Self::Int8),
            "int16" => Ok(Self::Int16),
            "int32" => Ok(Self::Int32),
            "int64" => Ok(Self::Int64),
            "uint8" => Ok(Self::UInt8),
            "uint16" => Ok(Self::UInt16),
            "uint32" => Ok(Self::UInt32),
            "uint64" => Ok(Self::UInt64),
            "float32" => Ok(Self::Float32),
            "float64" => Ok(Self::Float64),
            dtype => Err(PyValueError::new_err(format!(
                "unsupported dtype {dtype:?}; expected bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, or float64"
            ))),
        }
    }
}

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

    match DType::parse(dtype)? {
        DType::Bool => Err(PyValueError::new_err("arange does not support bool dtype")),
        DType::Int8 => output(py, NDArray::arange(start as i8, stop as i8, step as i8)),
        DType::Int16 => output(py, NDArray::arange(start as i16, stop as i16, step as i16)),
        DType::Int32 => output(py, NDArray::arange(start as i32, stop as i32, step as i32)),
        DType::Int64 => output(py, NDArray::arange(start as i64, stop as i64, step as i64)),
        DType::UInt8 => output(py, NDArray::arange(start as u8, stop as u8, step as u8)),
        DType::UInt16 => output(py, NDArray::arange(start as u16, stop as u16, step as u16)),
        DType::UInt32 => output(py, NDArray::arange(start as u32, stop as u32, step as u32)),
        DType::UInt64 => output(py, NDArray::arange(start as u64, stop as u64, step as u64)),
        DType::Float32 => output(py, NDArray::arange(start as f32, stop as f32, step as f32)),
        DType::Float64 => output(py, NDArray::arange(start, stop, step)),
    }
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
