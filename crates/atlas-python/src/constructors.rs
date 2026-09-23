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

pub(crate) fn eye(
    py: Python<'_>,
    rows: usize,
    columns: Option<usize>,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let columns = columns.unwrap_or(rows);

    match DType::parse(dtype)? {
        DType::Bool => Err(PyValueError::new_err("eye does not support bool dtype")),
        DType::Int8 => output(py, NDArray::<i8>::eye_with_columns(rows, columns)),
        DType::Int16 => output(py, NDArray::<i16>::eye_with_columns(rows, columns)),
        DType::Int32 => output(py, NDArray::<i32>::eye_with_columns(rows, columns)),
        DType::Int64 => output(py, NDArray::<i64>::eye_with_columns(rows, columns)),
        DType::UInt8 => output(py, NDArray::<u8>::eye_with_columns(rows, columns)),
        DType::UInt16 => output(py, NDArray::<u16>::eye_with_columns(rows, columns)),
        DType::UInt32 => output(py, NDArray::<u32>::eye_with_columns(rows, columns)),
        DType::UInt64 => output(py, NDArray::<u64>::eye_with_columns(rows, columns)),
        DType::Float32 => output(py, NDArray::<f32>::eye_with_columns(rows, columns)),
        DType::Float64 => output(py, NDArray::<f64>::eye_with_columns(rows, columns)),
    }
}

pub(crate) fn identity(py: Python<'_>, size: usize, dtype: Option<&str>) -> PyResult<Py<PyAny>> {
    eye(py, size, None, dtype)
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
    start: &Bound<'_, PyAny>,
    stop: Option<&Bound<'_, PyAny>>,
    step: Option<&Bound<'_, PyAny>>,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    macro_rules! integer_arange {
        ($ty:ty, $minimum:expr, $maximum_exclusive:expr) => {{
            let input_start = integer_arange_value(start, "start", $minimum, $maximum_exclusive)?;
            let start = if stop.is_some() { input_start } else { 0 };
            let stop = stop.map_or(Ok(input_start), |stop| {
                integer_arange_value(stop, "stop", $minimum, $maximum_exclusive)
            })?;
            let step = step.map_or(Ok(1), |step| {
                integer_arange_value(step, "step", $minimum, $maximum_exclusive)
            })?;
            let start = <$ty>::try_from(start).expect("integer arange bounds are validated");
            let stop = <$ty>::try_from(stop).expect("integer arange bounds are validated");
            let step = <$ty>::try_from(step).expect("integer arange bounds are validated");
            output(py, NDArray::arange(start, stop, step))
        }};
    }

    match DType::parse(dtype)? {
        DType::Bool => Err(PyValueError::new_err("arange does not support bool dtype")),
        DType::Int8 => integer_arange!(i8, i8::MIN as i128, i8::MAX as i128 + 1),
        DType::Int16 => integer_arange!(i16, i16::MIN as i128, i16::MAX as i128 + 1),
        DType::Int32 => integer_arange!(i32, i32::MIN as i128, i32::MAX as i128 + 1),
        DType::Int64 => integer_arange!(i64, i64::MIN as i128, i64::MAX as i128 + 1),
        DType::UInt8 => integer_arange!(u8, 0, u8::MAX as i128 + 1),
        DType::UInt16 => integer_arange!(u16, 0, u16::MAX as i128 + 1),
        DType::UInt32 => integer_arange!(u32, 0, u32::MAX as i128 + 1),
        DType::UInt64 => integer_arange!(u64, 0, u64::MAX as i128 + 1),
        DType::Float32 => floating_arange(py, start, stop, step, |start, stop, step| {
            NDArray::arange(start as f32, stop as f32, step as f32)
        }),
        DType::Float64 => floating_arange(py, start, stop, step, NDArray::arange),
    }
}

pub(crate) fn linspace(
    py: Python<'_>,
    start: f64,
    stop: f64,
    num: usize,
    dtype: Option<&str>,
    endpoint: bool,
) -> PyResult<Py<PyAny>> {
    match DType::parse(dtype)? {
        DType::Float32 => {
            output(py, NDArray::linspace_with_endpoint(start as f32, stop as f32, num, endpoint))
        }
        DType::Float64 => output(py, NDArray::linspace_with_endpoint(start, stop, num, endpoint)),
        _ => Err(PyValueError::new_err("linspace only supports float32 and float64 dtypes")),
    }
}

fn integer_arange_value(
    value: &Bound<'_, PyAny>,
    name: &str,
    minimum: i128,
    maximum_exclusive: i128,
) -> PyResult<i128> {
    if let Ok(value) = value.extract::<i128>() {
        if (minimum..maximum_exclusive).contains(&value) {
            return Ok(value);
        }
    } else {
        let value = value.extract::<f64>()?;
        if value.is_finite()
            && value.fract() == 0.0
            && value >= minimum as f64
            && value < maximum_exclusive as f64
        {
            return Ok(value as i128);
        }
    }

    Err(PyValueError::new_err(format!(
        "integer arange {name} must be a finite integral value within the target dtype range"
    )))
}

fn floating_arange<T>(
    py: Python<'_>,
    start: &Bound<'_, PyAny>,
    stop: Option<&Bound<'_, PyAny>>,
    step: Option<&Bound<'_, PyAny>>,
    arange: impl FnOnce(f64, f64, f64) -> atlas_ndarray::AtlasNdResult<NDArray<T>>,
) -> PyResult<Py<PyAny>>
where
    T: atlas_ndarray::ArrayElement + numpy::Element,
{
    let start = start.extract::<f64>()?;
    let stop = stop.map(|stop| stop.extract()).transpose()?;
    let step = step.map(|step| step.extract()).transpose()?;
    let (start, stop) = stop.map_or((0.0, start), |stop| (start, stop));

    output(py, arange(start, stop, step.unwrap_or(1.0)))
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
