use atlas_ndarray::NDArray;
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::array;

#[derive(Clone, Copy)]
enum DType {
    Bool,
    Int64,
    Float32,
    Float64,
}

impl DType {
    fn parse(dtype: Option<&str>) -> PyResult<Self> {
        match dtype.unwrap_or("float64") {
            "bool" => Ok(Self::Bool),
            "int64" => Ok(Self::Int64),
            "float32" => Ok(Self::Float32),
            "float64" => Ok(Self::Float64),
            dtype => Err(PyValueError::new_err(format!(
                "unsupported dtype {dtype:?}; expected bool, int64, float32, or float64"
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
        DType::Int64 => asarray_typed::<i64>(py, value),
        DType::Float32 => asarray_typed::<f32>(py, value),
        DType::Float64 => asarray_typed::<f64>(py, value),
    }
}

pub(crate) fn zeros(py: Python<'_>, shape: Vec<usize>, dtype: Option<&str>) -> PyResult<Py<PyAny>> {
    match DType::parse(dtype)? {
        DType::Bool => output(py, NDArray::full(shape, false)),
        DType::Int64 => output(py, NDArray::<i64>::zeros(shape)),
        DType::Float32 => output(py, NDArray::<f32>::zeros(shape)),
        DType::Float64 => output(py, NDArray::<f64>::zeros(shape)),
    }
}

pub(crate) fn ones(py: Python<'_>, shape: Vec<usize>, dtype: Option<&str>) -> PyResult<Py<PyAny>> {
    match DType::parse(dtype)? {
        DType::Bool => output(py, NDArray::full(shape, true)),
        DType::Int64 => output(py, NDArray::<i64>::ones(shape)),
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
        DType::Int64 => output(py, NDArray::full(shape, value.extract::<i64>()?)),
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
        DType::Int64 => output(py, NDArray::arange(start as i64, stop as i64, step as i64)),
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
    let array = array.map_err(|error| PyValueError::new_err(error.to_string()))?;

    output_owned(py, array)
}

fn output_owned<T>(py: Python<'_>, array: NDArray<T>) -> PyResult<Py<PyAny>>
where
    T: atlas_ndarray::ArrayElement + numpy::Element,
{
    Ok(array::to_numpy_owned(py, array)?.into_any().unbind())
}
