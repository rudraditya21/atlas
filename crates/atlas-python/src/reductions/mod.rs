//! Reduction Python binding modules.

#[path = "reduction.rs"]
mod functions;

pub(crate) use functions::{stddev, variance};
use pyo3::prelude::*;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::register_functions!(module; sum, mean, min, max, argmin, argmax, cumsum, cumprod, cumsum_axis, cumprod_axis, nanmin, nanmax, nanmean, nanstd, argmin_axis, argmax_axis, sum_axis, mean_axis, min_axis, max_axis);
    Ok(())
}

#[pyfunction(signature = (value, axis = None))]
fn sum(py: Python<'_>, value: &Bound<'_, PyAny>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
    functions::sum(py, value, axis)
}

#[pyfunction(signature = (value, axis = None))]
fn mean(py: Python<'_>, value: &Bound<'_, PyAny>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
    functions::mean(py, value, axis)
}

#[pyfunction(signature = (value, axis = None))]
fn min(py: Python<'_>, value: &Bound<'_, PyAny>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
    functions::min(py, value, axis)
}

#[pyfunction(signature = (value, axis = None))]
fn max(py: Python<'_>, value: &Bound<'_, PyAny>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
    functions::max(py, value, axis)
}

#[pyfunction(signature = (value, axis = None))]
fn argmin(py: Python<'_>, value: &Bound<'_, PyAny>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
    functions::argmin(py, value, axis)
}

#[pyfunction(signature = (value, axis = None))]
fn argmax(py: Python<'_>, value: &Bound<'_, PyAny>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
    functions::argmax(py, value, axis)
}

#[pyfunction(signature = (value, axis = None, *, out = None))]
fn cumsum(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    axis: Option<i64>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    functions::cumsum(py, value, axis, out)
}

#[pyfunction(signature = (value, axis = None, *, out = None))]
fn cumprod(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    axis: Option<i64>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    functions::cumprod(py, value, axis, out)
}

#[pyfunction(signature = (value, axis, *, out = None))]
fn cumsum_axis(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    axis: i64,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    functions::cumsum_axis(py, value, axis, out)
}

#[pyfunction(signature = (value, axis, *, out = None))]
fn cumprod_axis(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    axis: i64,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    functions::cumprod_axis(py, value, axis, out)
}

#[pyfunction]
fn nanmin(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    functions::nanmin(py, value)
}

#[pyfunction]
fn nanmax(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    functions::nanmax(py, value)
}

#[pyfunction]
fn nanmean(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    functions::nanmean(py, value)
}

#[pyfunction]
fn nanstd(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    functions::nanstd(py, value)
}

#[pyfunction]
fn argmin_axis(py: Python<'_>, value: &Bound<'_, PyAny>, axis: i64) -> PyResult<Py<PyAny>> {
    functions::argmin_axis(py, value, axis)
}

#[pyfunction]
fn argmax_axis(py: Python<'_>, value: &Bound<'_, PyAny>, axis: i64) -> PyResult<Py<PyAny>> {
    functions::argmax_axis(py, value, axis)
}

#[pyfunction(signature = (value, axis, *, out = None))]
fn sum_axis(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    axis: i64,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    functions::sum_axis(py, value, axis, out)
}

#[pyfunction(signature = (value, axis, *, out = None))]
fn mean_axis(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    axis: i64,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    functions::mean_axis(py, value, axis, out)
}

#[pyfunction(signature = (value, axis, *, out = None))]
fn min_axis(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    axis: i64,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    functions::min_axis(py, value, axis, out)
}

#[pyfunction(signature = (value, axis, *, out = None))]
fn max_axis(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    axis: i64,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    functions::max_axis(py, value, axis, out)
}
