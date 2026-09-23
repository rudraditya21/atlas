use atlas_ndarray::{ArrayElement, NDArray};
use numpy::Element;
use pyo3::prelude::*;

use crate::{array, gil, python_dtype::with_dtype};

macro_rules! with_array {
    ($py:expr, $value:expr, |$array:ident| $body:expr) => {{
        let dtype = array::source_dtype($py, $value)?;
        with_dtype!(
            dtype,
            all | T | {
                let $array = array::from_numpy(array::readonly_from_python::<T>($py, $value)?)?;
                $body
            }
        )
    }};
}

pub(crate) fn reshape(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    shape: Vec<usize>,
) -> PyResult<Py<PyAny>> {
    with_array!(py, value, |array| reshape_array(py, array, shape))
}

pub(crate) fn transpose(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    axes: Option<Vec<i64>>,
) -> PyResult<Py<PyAny>> {
    with_array!(py, value, |array| transpose_array(py, array, axes))
}

pub(crate) fn swap_axes(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    left: i64,
    right: i64,
) -> PyResult<Py<PyAny>> {
    with_array!(py, value, |array| swap_axes_array(py, array, left, right))
}

pub(crate) fn squeeze(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    axis: Option<i64>,
) -> PyResult<Py<PyAny>> {
    with_array!(py, value, |array| squeeze_array(py, array, axis))
}

pub(crate) fn expand_dims(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    axis: i64,
) -> PyResult<Py<PyAny>> {
    with_array!(py, value, |array| expand_dims_array(py, array, axis))
}

fn reshape_array<T>(py: Python<'_>, array: NDArray<T>, shape: Vec<usize>) -> PyResult<Py<PyAny>>
where
    T: ArrayElement + Element,
{
    let array = gil::without_gil(py, move || array.reshape(shape).map(|view| view.to_owned()))
        .map_err(|error| crate::error::ndarray(py, error))?;

    Ok(array::to_numpy_owned(py, array)?.into_any().unbind())
}

fn transpose_array<T>(
    py: Python<'_>,
    array: NDArray<T>,
    axes: Option<Vec<i64>>,
) -> PyResult<Py<PyAny>>
where
    T: ArrayElement + Element,
{
    let array = gil::without_gil(py, move || match axes {
        Some(axes) => array.permute_axes(axes).map(|view| view.to_owned()),
        None => Ok(array.view().transpose().to_owned()),
    })
    .map_err(|error| crate::error::ndarray(py, error))?;

    Ok(array::to_numpy_owned(py, array)?.into_any().unbind())
}

fn swap_axes_array<T>(
    py: Python<'_>,
    array: NDArray<T>,
    left: i64,
    right: i64,
) -> PyResult<Py<PyAny>>
where
    T: ArrayElement + Element,
{
    let array =
        gil::without_gil(py, move || array.swap_axes(left, right).map(|view| view.to_owned()))
            .map_err(|error| crate::error::ndarray(py, error))?;

    Ok(array::to_numpy_owned(py, array)?.into_any().unbind())
}

fn squeeze_array<T>(py: Python<'_>, array: NDArray<T>, axis: Option<i64>) -> PyResult<Py<PyAny>>
where
    T: ArrayElement + Element,
{
    let array = gil::without_gil(py, move || match axis {
        Some(axis) => array.squeeze_axis(axis).map(|view| view.to_owned()),
        None => Ok(array.squeeze().to_owned()),
    })
    .map_err(|error| crate::error::ndarray(py, error))?;

    Ok(array::to_numpy_owned(py, array)?.into_any().unbind())
}

fn expand_dims_array<T>(py: Python<'_>, array: NDArray<T>, axis: i64) -> PyResult<Py<PyAny>>
where
    T: ArrayElement + Element,
{
    let array = gil::without_gil(py, move || array.expand_dims(axis).map(|view| view.to_owned()))
        .map_err(|error| crate::error::ndarray(py, error))?;

    Ok(array::to_numpy_owned(py, array)?.into_any().unbind())
}
