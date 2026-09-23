use atlas_ndarray::NDArray;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PySequence},
};

use crate::{
    array,
    python_dtype::{DType, with_dtype},
};

pub(crate) fn asarray(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let source_dtype = array::source_dtype(py, value)?;
    let dtype = match dtype {
        Some(dtype) => DType::parse(py, Some(dtype))?,
        None => source_dtype,
    };

    if dtype == source_dtype {
        return Ok(value.clone().unbind());
    }

    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", dtype.name())?;
    Ok(PyModule::import(py, "numpy")?.getattr("asarray")?.call((value,), Some(&kwargs))?.unbind())
}

pub(crate) fn zeros(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let shape = shape_values(shape)?;
    let dtype = DType::parse(py, dtype)?;
    if matches!(dtype, DType::Bool) {
        return output(py, NDArray::full(shape, false));
    }

    with_dtype!(dtype, numeric | T | output(py, NDArray::<T>::zeros(shape)))
}

pub(crate) fn ones(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let shape = shape_values(shape)?;
    let dtype = DType::parse(py, dtype)?;
    if matches!(dtype, DType::Bool) {
        return output(py, NDArray::full(shape, true));
    }

    with_dtype!(dtype, numeric | T | output(py, NDArray::<T>::ones(shape)))
}

pub(crate) fn eye(
    py: Python<'_>,
    rows: usize,
    columns: Option<usize>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let columns = columns.unwrap_or(rows);

    let dtype = DType::parse(py, dtype)?;
    if matches!(dtype, DType::Bool) {
        return Err(PyValueError::new_err("eye does not support bool dtype"));
    }

    with_dtype!(dtype, numeric | T | output(py, NDArray::<T>::eye_with_columns(rows, columns)))
}

pub(crate) fn identity(
    py: Python<'_>,
    size: usize,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    eye(py, size, None, dtype)
}

pub(crate) fn full(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    fill_value: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let shape = shape_values(shape)?;
    let dtype = DType::parse(py, dtype)?;

    with_dtype!(dtype, all | T | output(py, NDArray::full(shape, fill_value.extract::<T>()?)))
}

fn shape_values(shape: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if shape.is_instance_of::<PySequence>() {
        return shape.extract();
    }

    shape.extract::<usize>().map(|size| vec![size])
}

pub(crate) fn arange(
    py: Python<'_>,
    start: &Bound<'_, PyAny>,
    stop: Option<&Bound<'_, PyAny>>,
    step: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
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

    match DType::parse(py, dtype)? {
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
    dtype: Option<&Bound<'_, PyAny>>,
    endpoint: bool,
) -> PyResult<Py<PyAny>> {
    match DType::parse(py, dtype)? {
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
