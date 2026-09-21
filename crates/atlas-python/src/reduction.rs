use atlas_ndarray::{ElementwiseArithmetic, NDArray, Numeric, RuntimeScalar};
use num_traits::ToPrimitive;
use numpy::Element;
use pyo3::{IntoPyObject, IntoPyObjectExt, exceptions::PyTypeError, prelude::*};

use crate::{array, gil};

#[derive(Clone, Copy)]
enum Reduction {
    Sum,
    Mean,
    Min,
    Max,
    Variance,
    Stddev,
}

#[derive(Clone, Copy)]
enum AxisReduction {
    Sum,
    Mean,
    Min,
    Max,
}

pub(crate) fn sum(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    reduce(py, value, Reduction::Sum)
}

pub(crate) fn mean(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    reduce(py, value, Reduction::Mean)
}

pub(crate) fn min(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    reduce(py, value, Reduction::Min)
}

pub(crate) fn max(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    reduce(py, value, Reduction::Max)
}

pub(crate) fn variance(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    reduce(py, value, Reduction::Variance)
}

pub(crate) fn stddev(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    reduce(py, value, Reduction::Stddev)
}

pub(crate) fn sum_axis(py: Python<'_>, value: &Bound<'_, PyAny>, axis: i64) -> PyResult<Py<PyAny>> {
    reduce_axis(py, value, axis, AxisReduction::Sum)
}

pub(crate) fn mean_axis(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    axis: i64,
) -> PyResult<Py<PyAny>> {
    reduce_axis(py, value, axis, AxisReduction::Mean)
}

pub(crate) fn min_axis(py: Python<'_>, value: &Bound<'_, PyAny>, axis: i64) -> PyResult<Py<PyAny>> {
    reduce_axis(py, value, axis, AxisReduction::Min)
}

pub(crate) fn max_axis(py: Python<'_>, value: &Bound<'_, PyAny>, axis: i64) -> PyResult<Py<PyAny>> {
    reduce_axis(py, value, axis, AxisReduction::Max)
}

fn reduce(py: Python<'_>, value: &Bound<'_, PyAny>, reduction: Reduction) -> PyResult<Py<PyAny>> {
    array::require_numpy_array(py, value)?;
    let dtype: String = value.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! reduce_from {
        ($ty:ty) => {{
            let array = array::from_numpy(array::readonly_from_python::<$ty>(py, value)?)?;
            match reduction {
                Reduction::Sum => reduce_sum(py, array),
                Reduction::Mean => reduce_mean(py, array),
                Reduction::Min => reduce_min(py, array),
                Reduction::Max => reduce_max(py, array),
                Reduction::Variance => reduce_variance(py, array),
                Reduction::Stddev => reduce_stddev(py, array),
            }
        }};
    }

    match dtype.as_str() {
        "int8" => reduce_from!(i8),
        "int16" => reduce_from!(i16),
        "int32" => reduce_from!(i32),
        "int64" => reduce_from!(i64),
        "uint8" => reduce_from!(u8),
        "uint16" => reduce_from!(u16),
        "uint32" => reduce_from!(u32),
        "uint64" => reduce_from!(u64),
        "float32" => reduce_from!(f32),
        "float64" => reduce_from!(f64),
        _ => Err(PyTypeError::new_err("reductions require a supported numeric NumPy dtype")),
    }
}

fn reduce_axis(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    axis: i64,
    reduction: AxisReduction,
) -> PyResult<Py<PyAny>> {
    array::require_numpy_array(py, value)?;
    let dtype: String = value.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! reduce_from {
        ($ty:ty) => {{
            let array = array::from_numpy(array::readonly_from_python::<$ty>(py, value)?)?;
            match reduction {
                AxisReduction::Sum => reduce_sum_axis(py, array, axis),
                AxisReduction::Mean => reduce_mean_axis(py, array, axis),
                AxisReduction::Min => reduce_min_axis(py, array, axis),
                AxisReduction::Max => reduce_max_axis(py, array, axis),
            }
        }};
    }

    match dtype.as_str() {
        "int8" => reduce_from!(i8),
        "int16" => reduce_from!(i16),
        "int32" => reduce_from!(i32),
        "int64" => reduce_from!(i64),
        "uint8" => reduce_from!(u8),
        "uint16" => reduce_from!(u16),
        "uint32" => reduce_from!(u32),
        "uint64" => reduce_from!(u64),
        "float32" => reduce_from!(f32),
        "float64" => reduce_from!(f64),
        _ => Err(PyTypeError::new_err("reductions require a supported numeric NumPy dtype")),
    }
}

fn reduce_sum<T>(py: Python<'_>, array: NDArray<T>) -> PyResult<Py<PyAny>>
where
    T: Numeric + ElementwiseArithmetic + RuntimeScalar + Element,
    for<'py> T: IntoPyObject<'py>,
{
    scalar(py, gil::without_gil(py, move || array.sum()))
}

fn reduce_mean<T>(py: Python<'_>, array: NDArray<T>) -> PyResult<Py<PyAny>>
where
    T: Numeric + ToPrimitive + Element,
{
    scalar(py, gil::without_gil(py, move || array.mean()))
}

fn reduce_min<T>(py: Python<'_>, array: NDArray<T>) -> PyResult<Py<PyAny>>
where
    T: Numeric + PartialOrd + RuntimeScalar + Element,
    for<'py> T: IntoPyObject<'py>,
{
    scalar(py, gil::without_gil(py, move || array.min()))
}

fn reduce_max<T>(py: Python<'_>, array: NDArray<T>) -> PyResult<Py<PyAny>>
where
    T: Numeric + PartialOrd + RuntimeScalar + Element,
    for<'py> T: IntoPyObject<'py>,
{
    scalar(py, gil::without_gil(py, move || array.max()))
}

fn reduce_variance<T>(py: Python<'_>, array: NDArray<T>) -> PyResult<Py<PyAny>>
where
    T: Numeric + ToPrimitive + Element,
{
    scalar(py, gil::without_gil(py, move || array.variance()))
}

fn reduce_stddev<T>(py: Python<'_>, array: NDArray<T>) -> PyResult<Py<PyAny>>
where
    T: Numeric + ToPrimitive + Element,
{
    scalar(py, gil::without_gil(py, move || array.stddev()))
}

fn reduce_sum_axis<T>(py: Python<'_>, array: NDArray<T>, axis: i64) -> PyResult<Py<PyAny>>
where
    T: Numeric + ElementwiseArithmetic + Element,
{
    array_output(py, gil::without_gil(py, move || array.sum_axis(axis)))
}

fn reduce_mean_axis<T>(py: Python<'_>, array: NDArray<T>, axis: i64) -> PyResult<Py<PyAny>>
where
    T: Numeric + ToPrimitive + Element,
{
    array_output(py, gil::without_gil(py, move || array.mean_axis(axis)))
}

fn reduce_min_axis<T>(py: Python<'_>, array: NDArray<T>, axis: i64) -> PyResult<Py<PyAny>>
where
    T: Numeric + PartialOrd + Element,
{
    array_output(py, gil::without_gil(py, move || array.min_axis(axis)))
}

fn reduce_max_axis<T>(py: Python<'_>, array: NDArray<T>, axis: i64) -> PyResult<Py<PyAny>>
where
    T: Numeric + PartialOrd + Element,
{
    array_output(py, gil::without_gil(py, move || array.max_axis(axis)))
}

fn scalar<T>(py: Python<'_>, result: atlas_ndarray::AtlasNdResult<T>) -> PyResult<Py<PyAny>>
where
    for<'py> T: IntoPyObject<'py>,
{
    result.map_err(|error| crate::error::ndarray(py, error))?.into_py_any(py)
}

fn array_output<T>(
    py: Python<'_>,
    result: atlas_ndarray::AtlasNdResult<NDArray<T>>,
) -> PyResult<Py<PyAny>>
where
    T: atlas_ndarray::ArrayElement + Element,
{
    Ok(array::to_numpy_owned(py, result.map_err(|error| crate::error::ndarray(py, error))?)?
        .into_any()
        .unbind())
}
