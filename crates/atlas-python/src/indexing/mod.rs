//! Indexing Python binding modules.

#[path = "argpartition.rs"]
mod argpartition_ops;
#[path = "argsort.rs"]
mod argsort_ops;
#[path = "partition.rs"]
mod partition_ops;
#[path = "put.rs"]
mod put_ops;
#[path = "searchsorted.rs"]
mod searchsorted_ops;
#[path = "sort.rs"]
mod sort_ops;
#[path = "take.rs"]
mod take_ops;
#[path = "unique.rs"]
mod unique_ops;

use pyo3::prelude::*;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::register_functions!(module; take, sort, argsort, argpartition, unique, searchsorted, partition, put);
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
