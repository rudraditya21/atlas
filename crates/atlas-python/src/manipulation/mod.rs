//! Manipulation Python binding modules.

#[path = "concat.rs"]
mod concat_ops;
#[path = "flatten.rs"]
mod flatten_ops;
#[path = "flip.rs"]
mod flip_ops;
#[path = "pad.rs"]
mod pad_ops;
#[path = "ravel.rs"]
mod ravel_ops;
#[path = "repeat.rs"]
mod repeat_ops;
#[path = "roll.rs"]
mod roll_ops;
#[path = "shape_ops.rs"]
mod shape_ops;
#[path = "split.rs"]
mod split_ops;
#[path = "stack.rs"]
mod stack_ops;
#[path = "tile.rs"]
mod tile_ops;

use pyo3::prelude::*;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::register_functions!(module; concatenate, stack, ravel, flatten, reshape, transpose, moveaxis, swap_axes, split, repeat, tile, flip, roll, pad, squeeze, expand_dims);
    Ok(())
}

#[pyfunction(signature = (arrays, axis = 0, *, out = None))]
fn concatenate(
    py: Python<'_>,
    arrays: Vec<Py<PyAny>>,
    axis: i64,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    concat_ops::concatenate(py, arrays, axis, out)
}

#[pyfunction(signature = (arrays, axis = 0, *, out = None))]
fn stack(
    py: Python<'_>,
    arrays: Vec<Py<PyAny>>,
    axis: i64,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    stack_ops::stack(py, arrays, axis, out)
}

#[pyfunction]
fn ravel(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    ravel_ops::ravel(py, value)
}

#[pyfunction]
fn flatten(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    flatten_ops::flatten(py, value)
}

#[pyfunction]
fn reshape(py: Python<'_>, value: &Bound<'_, PyAny>, shape: Vec<i128>) -> PyResult<Py<PyAny>> {
    shape_ops::reshape(py, value, shape)
}

#[pyfunction(signature = (value, axes = None))]
fn transpose(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    axes: Option<Vec<i64>>,
) -> PyResult<Py<PyAny>> {
    shape_ops::transpose(py, value, axes)
}

#[pyfunction]
fn moveaxis(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    source: &Bound<'_, PyAny>,
    destination: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    shape_ops::moveaxis(py, value, source, destination)
}

#[pyfunction]
fn swap_axes(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    left: i64,
    right: i64,
) -> PyResult<Py<PyAny>> {
    shape_ops::swap_axes(py, value, left, right)
}

#[pyfunction]
fn split(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    indices_or_sections: &Bound<'_, PyAny>,
    axis: i64,
) -> PyResult<Vec<Py<PyAny>>> {
    split_ops::split(py, value, indices_or_sections, axis)
}

#[pyfunction(signature = (value, repeats, axis = None))]
fn repeat(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    repeats: &Bound<'_, PyAny>,
    axis: Option<i64>,
) -> PyResult<Py<PyAny>> {
    repeat_ops::repeat(py, value, repeats, axis)
}

#[pyfunction]
fn tile(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    repetitions: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    tile_ops::tile(py, value, repetitions)
}

#[pyfunction(signature = (value, axis = None))]
fn flip(py: Python<'_>, value: &Bound<'_, PyAny>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
    flip_ops::flip(py, value, axis)
}

#[pyfunction(signature = (value, shift, axis = None))]
fn roll(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    shift: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    roll_ops::roll(py, value, shift, axis)
}

#[pyfunction(signature = (array, widths, value = None))]
fn pad(
    py: Python<'_>,
    array: &Bound<'_, PyAny>,
    widths: &Bound<'_, PyAny>,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    pad_ops::pad(py, array, widths, value)
}

#[pyfunction(signature = (value, axis = None))]
fn squeeze(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    axis: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    shape_ops::squeeze(py, value, axis)
}

#[pyfunction]
fn expand_dims(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    axis: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    shape_ops::expand_dims(py, value, axis)
}
