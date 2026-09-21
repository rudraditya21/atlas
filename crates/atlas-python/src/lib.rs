//! Native Python bindings for Atlas.
//!
//! This crate owns the Python extension boundary. Numerical implementations remain in the
//! Atlas library crates and will be registered here incrementally.

use pyo3::prelude::*;

mod arithmetic;
mod array;
mod casting;
#[path = "clip.rs"]
mod clip_ops;
mod close;
mod constructors;
mod error;
mod gil;
mod logical;
#[path = "matmul.rs"]
mod matmul_ops;
mod metadata;
#[path = "dtype.rs"]
mod python_dtype;
mod reduction;
#[cfg(feature = "test-support")]
mod scalar;
mod shape_ops;
#[cfg(feature = "test-support")]
mod test_support;
mod unary;
mod where_ops;

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add_function(wrap_pyfunction!(asarray, module)?)?;
    module.add_function(wrap_pyfunction!(zeros, module)?)?;
    module.add_function(wrap_pyfunction!(ones, module)?)?;
    module.add_function(wrap_pyfunction!(full, module)?)?;
    module.add_function(wrap_pyfunction!(arange, module)?)?;
    module.add_function(wrap_pyfunction!(astype, module)?)?;
    module.add_function(wrap_pyfunction!(shape, module)?)?;
    module.add_function(wrap_pyfunction!(ndim, module)?)?;
    module.add_function(wrap_pyfunction!(size, module)?)?;
    module.add_function(wrap_pyfunction!(dtype, module)?)?;
    module.add_function(wrap_pyfunction!(add, module)?)?;
    module.add_function(wrap_pyfunction!(subtract, module)?)?;
    module.add_function(wrap_pyfunction!(multiply, module)?)?;
    module.add_function(wrap_pyfunction!(divide, module)?)?;
    module.add_function(wrap_pyfunction!(equal, module)?)?;
    module.add_function(wrap_pyfunction!(not_equal, module)?)?;
    module.add_function(wrap_pyfunction!(less, module)?)?;
    module.add_function(wrap_pyfunction!(less_equal, module)?)?;
    module.add_function(wrap_pyfunction!(greater, module)?)?;
    module.add_function(wrap_pyfunction!(greater_equal, module)?)?;
    module.add_function(wrap_pyfunction!(select, module)?)?;
    module.add_function(wrap_pyfunction!(count_true, module)?)?;
    module.add_function(wrap_pyfunction!(all, module)?)?;
    module.add_function(wrap_pyfunction!(any, module)?)?;
    module.add_function(wrap_pyfunction!(all_axis, module)?)?;
    module.add_function(wrap_pyfunction!(any_axis, module)?)?;
    module.add_function(wrap_pyfunction!(where_, module)?)?;
    module.add_function(wrap_pyfunction!(nonzero, module)?)?;
    module.add_function(wrap_pyfunction!(masked_fill, module)?)?;
    module.add_function(wrap_pyfunction!(sum, module)?)?;
    module.add_function(wrap_pyfunction!(mean, module)?)?;
    module.add_function(wrap_pyfunction!(min, module)?)?;
    module.add_function(wrap_pyfunction!(max, module)?)?;
    module.add_function(wrap_pyfunction!(variance, module)?)?;
    module.add_function(wrap_pyfunction!(stddev, module)?)?;
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
    module.add_function(wrap_pyfunction!(allclose, module)?)?;
    module.add_function(wrap_pyfunction!(clip, module)?)?;
    module.add_function(wrap_pyfunction!(matmul, module)?)?;
    module.add_function(wrap_pyfunction!(neg, module)?)?;
    module.add_function(wrap_pyfunction!(abs, module)?)?;
    module.add_function(wrap_pyfunction!(sign, module)?)?;
    module.add_function(wrap_pyfunction!(round, module)?)?;
    module.add_function(wrap_pyfunction!(isnan, module)?)?;
    module.add_function(wrap_pyfunction!(isinf, module)?)?;
    module.add_function(wrap_pyfunction!(isfinite, module)?)?;

    #[cfg(feature = "test-support")]
    test_support::register(module)?;

    Ok(())
}

#[pyfunction(signature = (value, dtype = None))]
fn asarray(py: Python<'_>, value: &Bound<'_, PyAny>, dtype: Option<&str>) -> PyResult<Py<PyAny>> {
    constructors::asarray(py, value, dtype)
}

#[pyfunction(signature = (shape, dtype = None))]
fn zeros(py: Python<'_>, shape: Vec<usize>, dtype: Option<&str>) -> PyResult<Py<PyAny>> {
    constructors::zeros(py, shape, dtype)
}

#[pyfunction(signature = (shape, dtype = None))]
fn ones(py: Python<'_>, shape: Vec<usize>, dtype: Option<&str>) -> PyResult<Py<PyAny>> {
    constructors::ones(py, shape, dtype)
}

#[pyfunction(signature = (shape, value, dtype = None))]
fn full(
    py: Python<'_>,
    shape: Vec<usize>,
    value: &Bound<'_, PyAny>,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    constructors::full(py, shape, value, dtype)
}

#[pyfunction(signature = (start, stop = None, step = None, dtype = None))]
fn arange(
    py: Python<'_>,
    start: f64,
    stop: Option<f64>,
    step: Option<f64>,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    constructors::arange(py, start, stop, step, dtype)
}

#[pyfunction]
fn astype(py: Python<'_>, value: &Bound<'_, PyAny>, dtype: &str) -> PyResult<Py<PyAny>> {
    casting::astype(py, value, dtype)
}

#[pyfunction]
fn shape(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    metadata::shape(py, value)
}

#[pyfunction]
fn ndim(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<usize> {
    metadata::ndim(py, value)
}

#[pyfunction]
fn size(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<usize> {
    metadata::size(py, value)
}

#[pyfunction]
fn dtype(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    metadata::dtype(py, value)
}

#[pyfunction]
fn add(py: Python<'_>, lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    arithmetic::add(py, lhs, rhs)
}

#[pyfunction]
fn subtract(py: Python<'_>, lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    arithmetic::subtract(py, lhs, rhs)
}

#[pyfunction]
fn multiply(py: Python<'_>, lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    arithmetic::multiply(py, lhs, rhs)
}

#[pyfunction]
fn divide(py: Python<'_>, lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    arithmetic::divide(py, lhs, rhs)
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

#[pyfunction]
fn all(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    logical::all(py, value)
}

#[pyfunction]
fn any(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    logical::any(py, value)
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
fn nonzero(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    logical::nonzero(py, value)
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

#[pyfunction]
fn sum(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    reduction::sum(py, value)
}

#[pyfunction]
fn mean(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    reduction::mean(py, value)
}

#[pyfunction]
fn min(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    reduction::min(py, value)
}

#[pyfunction]
fn max(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    reduction::max(py, value)
}

#[pyfunction]
fn variance(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    reduction::variance(py, value)
}

#[pyfunction]
fn stddev(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    reduction::stddev(py, value)
}

#[pyfunction]
fn argmin(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    reduction::argmin(py, value)
}

#[pyfunction]
fn argmax(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    reduction::argmax(py, value)
}

#[pyfunction]
fn cumsum(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    reduction::cumsum(py, value)
}

#[pyfunction]
fn cumprod(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    reduction::cumprod(py, value)
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
fn reshape(py: Python<'_>, value: &Bound<'_, PyAny>, shape: Vec<usize>) -> PyResult<Py<PyAny>> {
    shape_ops::reshape(py, value, shape)
}

#[pyfunction]
fn transpose(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    shape_ops::transpose(py, value)
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

#[pyfunction]
fn clip(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    minimum: &Bound<'_, PyAny>,
    maximum: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    clip_ops::clip(py, value, minimum, maximum)
}

#[pyfunction]
fn matmul(py: Python<'_>, lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    matmul_ops::matmul(py, lhs, rhs)
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

#[pyfunction]
fn round(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    unary::round(py, value)
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
