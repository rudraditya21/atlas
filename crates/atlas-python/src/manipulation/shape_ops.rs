use atlas_ndarray::{ArrayElement, AtlasNdError, AtlasNdResult, NDArray};
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
    shape: Vec<i128>,
) -> PyResult<Py<PyAny>> {
    with_array!(py, value, |array| {
        let shape = resolve_reshape_shape(&array, shape)
            .map_err(|error| crate::error::ndarray(py, error))?;
        reshape_array(py, array, shape)
    })
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

fn resolve_reshape_shape<T>(array: &NDArray<T>, shape: Vec<i128>) -> AtlasNdResult<Vec<usize>>
where
    T: ArrayElement,
{
    let mut inferred = None;
    let mut known_size = 1_usize;
    let mut shape = shape
        .into_iter()
        .enumerate()
        .map(|(index, dimension)| match dimension {
            -1 => {
                if inferred.replace(index).is_some() {
                    return Err(reshape_error("shape can contain only one inferred dimension"));
                }
                Ok(0)
            }
            dimension if dimension < 0 => {
                Err(reshape_error("shape dimensions must be nonnegative or -1"))
            }
            dimension => {
                let dimension =
                    usize::try_from(dimension).map_err(|_| reshape_error("shape overflow"))?;
                known_size = known_size
                    .checked_mul(dimension)
                    .ok_or_else(|| reshape_error("shape overflow"))?;
                Ok(dimension)
            }
        })
        .collect::<AtlasNdResult<Vec<_>>>()?;

    if let Some(index) = inferred {
        if known_size == 0 {
            return Err(reshape_error("cannot infer a dimension with zero-sized known dimensions"));
        }
        if array.len() % known_size != 0 {
            return Err(reshape_error("known dimensions must divide the array size"));
        }
        shape[index] = array.len() / known_size;
    }

    Ok(shape)
}

fn reshape_error(reason: &'static str) -> AtlasNdError {
    AtlasNdError::InvalidArgument { op: "reshape", reason }
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
