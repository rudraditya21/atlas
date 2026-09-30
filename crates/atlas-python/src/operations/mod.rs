//! Operation Python binding modules.

#[path = "arithmetic.rs"]
mod arithmetic;
#[path = "bitwise.rs"]
mod bitwise;
#[path = "clip.rs"]
mod clip_ops;
#[path = "close.rs"]
mod close;
#[path = "logical.rs"]
mod logical;
#[path = "unary.rs"]
mod unary;
#[path = "where_ops.rs"]
mod where_ops;

use pyo3::prelude::*;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::register_functions!(module; add, subtract, multiply, divide, equal, not_equal, less, less_equal, greater, greater_equal, select, count_true, all, any, all_axis, any_axis, where_, bitwise_and, bitwise_or, bitwise_xor, bitwise_not, nonzero, argwhere, masked_fill, allclose, clip, neg, abs, sign, round, isnan, isinf, isfinite);
    Ok(())
}

#[pyfunction(signature = (lhs, rhs, *, out = None))]
fn add(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    arithmetic::add(py, lhs, rhs, out)
}

#[pyfunction(signature = (lhs, rhs, *, out = None))]
fn subtract(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    arithmetic::subtract(py, lhs, rhs, out)
}

#[pyfunction(signature = (lhs, rhs, *, out = None))]
fn multiply(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    arithmetic::multiply(py, lhs, rhs, out)
}

#[pyfunction(signature = (lhs, rhs, *, out = None))]
fn divide(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    arithmetic::divide(py, lhs, rhs, out)
}

#[pyfunction]
fn equal(py: Python<'_>, lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    logical::equal(py, lhs, rhs)
}

#[pyfunction]
fn not_equal(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    logical::not_equal(py, lhs, rhs)
}

#[pyfunction]
fn less(py: Python<'_>, lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    logical::less(py, lhs, rhs)
}

#[pyfunction]
fn less_equal(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    logical::less_equal(py, lhs, rhs)
}

#[pyfunction]
fn greater(py: Python<'_>, lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    logical::greater(py, lhs, rhs)
}

#[pyfunction]
fn greater_equal(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    logical::greater_equal(py, lhs, rhs)
}

#[pyfunction]
fn select(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    mask: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    logical::select(py, value, mask)
}

#[pyfunction]
fn count_true(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<usize> {
    logical::count_true(py, value)
}

#[pyfunction(signature = (value, axis = None))]
fn all(py: Python<'_>, value: &Bound<'_, PyAny>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
    logical::all(py, value, axis)
}

#[pyfunction(signature = (value, axis = None))]
fn any(py: Python<'_>, value: &Bound<'_, PyAny>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
    logical::any(py, value, axis)
}

#[pyfunction]
fn all_axis(py: Python<'_>, value: &Bound<'_, PyAny>, axis: i64) -> PyResult<Py<PyAny>> {
    logical::all_axis(py, value, axis)
}

#[pyfunction]
fn any_axis(py: Python<'_>, value: &Bound<'_, PyAny>, axis: i64) -> PyResult<Py<PyAny>> {
    logical::any_axis(py, value, axis)
}

#[pyfunction(name = "where")]
fn where_(
    py: Python<'_>,
    condition: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    where_ops::where_(py, condition, x, y)
}

#[pyfunction]
fn bitwise_and(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    bitwise::bitwise_and(py, lhs, rhs)
}

#[pyfunction]
fn bitwise_or(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    bitwise::bitwise_or(py, lhs, rhs)
}

#[pyfunction]
fn bitwise_xor(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    bitwise::bitwise_xor(py, lhs, rhs)
}

#[pyfunction]
fn bitwise_not(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    bitwise::bitwise_not(py, value)
}

#[pyfunction(signature = (lhs, rhs, rtol = 1e-5, atol = 1e-8, equal_nan = false))]
fn allclose(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> PyResult<bool> {
    close::allclose(py, lhs, rhs, rtol, atol, equal_nan)
}

#[pyfunction(signature = (value, minimum = None, maximum = None))]
fn clip(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    minimum: Option<&Bound<'_, PyAny>>,
    maximum: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    clip_ops::clip(py, value, minimum, maximum)
}

#[pyfunction]
fn neg(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    unary::neg(py, value)
}

#[pyfunction]
fn abs(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    unary::abs(py, value)
}

#[pyfunction]
fn sign(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    unary::sign(py, value)
}

#[pyfunction(signature = (value, decimals = 0))]
fn round(py: Python<'_>, value: &Bound<'_, PyAny>, decimals: i64) -> PyResult<Py<PyAny>> {
    unary::round(py, value, decimals)
}

#[pyfunction]
fn isnan(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    unary::isnan(py, value)
}

#[pyfunction]
fn isinf(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    unary::isinf(py, value)
}

#[pyfunction]
fn isfinite(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    unary::isfinite(py, value)
}

#[pyfunction]
fn nonzero(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    logical::nonzero(py, value)
}

#[pyfunction]
fn argwhere(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    logical::argwhere(py, value)
}

#[pyfunction]
fn masked_fill(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    mask: &Bound<'_, PyAny>,
    fill: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    logical::masked_fill(py, value, mask, fill)
}
