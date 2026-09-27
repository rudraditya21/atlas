//! Arrays Python bindings.

use pyo3::{prelude::*, wrap_pyfunction};

use crate::{
    argpartition_ops, argsort_ops, concat_ops, flatten_ops, flip_ops, pad_ops, partition_ops,
    put_ops, ravel_ops, reduction, repeat_ops, roll_ops, searchsorted_ops, shape_ops, sort_ops,
    split_ops, stack_ops, take_ops, tile_ops, unique_ops,
};

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(take, module)?)?;
    module.add_function(wrap_pyfunction!(concatenate, module)?)?;
    module.add_function(wrap_pyfunction!(stack, module)?)?;
    module.add_function(wrap_pyfunction!(ravel, module)?)?;
    module.add_function(wrap_pyfunction!(flatten, module)?)?;
    module.add_function(wrap_pyfunction!(sum, module)?)?;
    module.add_function(wrap_pyfunction!(mean, module)?)?;
    module.add_function(wrap_pyfunction!(min, module)?)?;
    module.add_function(wrap_pyfunction!(max, module)?)?;
    module.add_function(wrap_pyfunction!(argmin, module)?)?;
    module.add_function(wrap_pyfunction!(argmax, module)?)?;
    module.add_function(wrap_pyfunction!(cumsum, module)?)?;
    module.add_function(wrap_pyfunction!(cumprod, module)?)?;
    module.add_function(wrap_pyfunction!(cumsum_axis, module)?)?;
    module.add_function(wrap_pyfunction!(cumprod_axis, module)?)?;
    module.add_function(wrap_pyfunction!(nanmin, module)?)?;
    module.add_function(wrap_pyfunction!(nanmax, module)?)?;
    module.add_function(wrap_pyfunction!(nanmean, module)?)?;
    module.add_function(wrap_pyfunction!(nanstd, module)?)?;
    module.add_function(wrap_pyfunction!(argmin_axis, module)?)?;
    module.add_function(wrap_pyfunction!(argmax_axis, module)?)?;
    module.add_function(wrap_pyfunction!(sum_axis, module)?)?;
    module.add_function(wrap_pyfunction!(mean_axis, module)?)?;
    module.add_function(wrap_pyfunction!(min_axis, module)?)?;
    module.add_function(wrap_pyfunction!(max_axis, module)?)?;
    module.add_function(wrap_pyfunction!(reshape, module)?)?;
    module.add_function(wrap_pyfunction!(transpose, module)?)?;
    module.add_function(wrap_pyfunction!(moveaxis, module)?)?;
    module.add_function(wrap_pyfunction!(swap_axes, module)?)?;
    module.add_function(wrap_pyfunction!(split, module)?)?;
    module.add_function(wrap_pyfunction!(repeat, module)?)?;
    module.add_function(wrap_pyfunction!(tile, module)?)?;
    module.add_function(wrap_pyfunction!(flip, module)?)?;
    module.add_function(wrap_pyfunction!(roll, module)?)?;
    module.add_function(wrap_pyfunction!(sort, module)?)?;
    module.add_function(wrap_pyfunction!(argsort, module)?)?;
    module.add_function(wrap_pyfunction!(argpartition, module)?)?;
    module.add_function(wrap_pyfunction!(unique, module)?)?;
    module.add_function(wrap_pyfunction!(pad, module)?)?;
    module.add_function(wrap_pyfunction!(searchsorted, module)?)?;
    module.add_function(wrap_pyfunction!(partition, module)?)?;
    module.add_function(wrap_pyfunction!(put, module)?)?;
    module.add_function(wrap_pyfunction!(squeeze, module)?)?;
    module.add_function(wrap_pyfunction!(expand_dims, module)?)?;
    Ok(())
}

#[pyfunction(signature = (value, indices, axis = None))]
fn take(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    indices: Vec<i64>,
    axis: Option<i64>,
) -> PyResult<Py<PyAny>> {
    take_ops::take(py, value, indices, axis)
}

#[pyfunction(signature = (arrays, axis = 0))]
fn concatenate(py: Python<'_>, arrays: Vec<Py<PyAny>>, axis: i64) -> PyResult<Py<PyAny>> {
    concat_ops::concatenate(py, arrays, axis)
}

#[pyfunction(signature = (arrays, axis = 0))]
fn stack(py: Python<'_>, arrays: Vec<Py<PyAny>>, axis: i64) -> PyResult<Py<PyAny>> {
    stack_ops::stack(py, arrays, axis)
}

#[pyfunction]
fn ravel(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    ravel_ops::ravel(py, value)
}

#[pyfunction]
fn flatten(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    flatten_ops::flatten(py, value)
}

#[pyfunction(signature = (value, axis = None))]
fn sum(py: Python<'_>, value: &Bound<'_, PyAny>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
    reduction::sum(py, value, axis)
}

#[pyfunction(signature = (value, axis = None))]
fn mean(py: Python<'_>, value: &Bound<'_, PyAny>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
    reduction::mean(py, value, axis)
}

#[pyfunction(signature = (value, axis = None))]
fn min(py: Python<'_>, value: &Bound<'_, PyAny>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
    reduction::min(py, value, axis)
}

#[pyfunction(signature = (value, axis = None))]
fn max(py: Python<'_>, value: &Bound<'_, PyAny>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
    reduction::max(py, value, axis)
}

#[pyfunction(signature = (value, axis = None))]
fn argmin(py: Python<'_>, value: &Bound<'_, PyAny>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
    reduction::argmin(py, value, axis)
}

#[pyfunction(signature = (value, axis = None))]
fn argmax(py: Python<'_>, value: &Bound<'_, PyAny>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
    reduction::argmax(py, value, axis)
}

#[pyfunction(signature = (value, axis = None))]
fn cumsum(py: Python<'_>, value: &Bound<'_, PyAny>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
    reduction::cumsum(py, value, axis)
}

#[pyfunction(signature = (value, axis = None))]
fn cumprod(py: Python<'_>, value: &Bound<'_, PyAny>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
    reduction::cumprod(py, value, axis)
}

#[pyfunction]
fn cumsum_axis(py: Python<'_>, value: &Bound<'_, PyAny>, axis: i64) -> PyResult<Py<PyAny>> {
    reduction::cumsum_axis(py, value, axis)
}

#[pyfunction]
fn cumprod_axis(py: Python<'_>, value: &Bound<'_, PyAny>, axis: i64) -> PyResult<Py<PyAny>> {
    reduction::cumprod_axis(py, value, axis)
}

#[pyfunction]
fn nanmin(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    reduction::nanmin(py, value)
}

#[pyfunction]
fn nanmax(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    reduction::nanmax(py, value)
}

#[pyfunction]
fn nanmean(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    reduction::nanmean(py, value)
}

#[pyfunction]
fn nanstd(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    reduction::nanstd(py, value)
}

#[pyfunction]
fn argmin_axis(py: Python<'_>, value: &Bound<'_, PyAny>, axis: i64) -> PyResult<Py<PyAny>> {
    reduction::argmin_axis(py, value, axis)
}

#[pyfunction]
fn argmax_axis(py: Python<'_>, value: &Bound<'_, PyAny>, axis: i64) -> PyResult<Py<PyAny>> {
    reduction::argmax_axis(py, value, axis)
}

#[pyfunction]
fn sum_axis(py: Python<'_>, value: &Bound<'_, PyAny>, axis: i64) -> PyResult<Py<PyAny>> {
    reduction::sum_axis(py, value, axis)
}

#[pyfunction]
fn mean_axis(py: Python<'_>, value: &Bound<'_, PyAny>, axis: i64) -> PyResult<Py<PyAny>> {
    reduction::mean_axis(py, value, axis)
}

#[pyfunction]
fn min_axis(py: Python<'_>, value: &Bound<'_, PyAny>, axis: i64) -> PyResult<Py<PyAny>> {
    reduction::min_axis(py, value, axis)
}

#[pyfunction]
fn max_axis(py: Python<'_>, value: &Bound<'_, PyAny>, axis: i64) -> PyResult<Py<PyAny>> {
    reduction::max_axis(py, value, axis)
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

#[pyfunction(signature = (value, axis = -1))]
fn sort(py: Python<'_>, value: &Bound<'_, PyAny>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
    sort_ops::sort(py, value, axis)
}

#[pyfunction(signature = (value, axis = -1))]
fn argsort(py: Python<'_>, value: &Bound<'_, PyAny>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
    argsort_ops::argsort(py, value, axis)
}

#[pyfunction(signature = (value, kth, axis = -1))]
fn argpartition(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    kth: &Bound<'_, PyAny>,
    axis: Option<i64>,
) -> PyResult<Py<PyAny>> {
    argpartition_ops::argpartition(py, value, kth, axis)
}

#[pyfunction(signature = (value, return_index = false, return_inverse = false, return_counts = false))]
fn unique(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    return_index: bool,
    return_inverse: bool,
    return_counts: bool,
) -> PyResult<Py<PyAny>> {
    unique_ops::unique(py, value, return_index, return_inverse, return_counts)
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

#[pyfunction(signature = (sorted, values, side = "left", sorter = None))]
fn searchsorted(
    py: Python<'_>,
    sorted: &Bound<'_, PyAny>,
    values: &Bound<'_, PyAny>,
    side: &str,
    sorter: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    searchsorted_ops::searchsorted(py, sorted, values, side, sorter)
}

#[pyfunction(signature = (value, kth, axis = -1))]
fn partition(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    kth: &Bound<'_, PyAny>,
    axis: Option<i64>,
) -> PyResult<Py<PyAny>> {
    partition_ops::partition(py, value, kth, axis)
}

#[pyfunction]
fn put(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    indices: Vec<i64>,
    values: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    put_ops::put(py, value, indices, values)
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
