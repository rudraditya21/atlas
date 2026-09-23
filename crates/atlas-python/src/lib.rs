//! Native Python bindings for Atlas.
//!
//! This crate owns the Python extension boundary. Numerical implementations remain in the
//! Atlas library crates and will be registered here incrementally.

use pyo3::prelude::*;

#[path = "indexing/argpartition.rs"]
mod argpartition_ops;
#[path = "indexing/argsort.rs"]
mod argsort_ops;
#[path = "operations/arithmetic.rs"]
mod arithmetic;
#[path = "support/array.rs"]
mod array;
#[path = "operations/bitwise.rs"]
mod bitwise;
#[path = "constructors/casting.rs"]
mod casting;
#[path = "operations/clip.rs"]
mod clip_ops;
#[path = "operations/close.rs"]
mod close;
#[path = "manipulation/concat.rs"]
mod concat_ops;
#[path = "constructors/constructors.rs"]
mod constructors;
#[path = "linalg/determinant.rs"]
mod determinant_ops;
#[path = "linalg/diag.rs"]
mod diag_ops;
#[path = "linalg/dot.rs"]
mod dot_ops;
#[path = "support/error.rs"]
mod error;
#[path = "manipulation/flatten.rs"]
mod flatten_ops;
#[path = "manipulation/flip.rs"]
mod flip_ops;
#[path = "support/gil.rs"]
mod gil;
#[path = "linalg/inverse.rs"]
mod inverse_ops;
#[path = "operations/logical.rs"]
mod logical;
#[path = "linalg/matmul.rs"]
mod matmul_ops;
#[path = "linalg/matrix_norm.rs"]
mod matrix_norm_ops;
#[path = "support/metadata.rs"]
mod metadata;
#[path = "linalg/norm.rs"]
mod norm_ops;
#[path = "manipulation/pad.rs"]
mod pad_ops;
#[path = "indexing/partition.rs"]
mod partition_ops;
#[path = "support/dtype.rs"]
mod python_dtype;
#[path = "manipulation/ravel.rs"]
mod ravel_ops;
#[path = "reductions/reduction.rs"]
mod reduction;
#[path = "manipulation/repeat.rs"]
mod repeat_ops;
#[path = "manipulation/roll.rs"]
mod roll_ops;
#[cfg(feature = "test-support")]
#[path = "support/scalar.rs"]
mod scalar;
#[path = "indexing/searchsorted.rs"]
mod searchsorted_ops;
#[path = "manipulation/shape_ops.rs"]
mod shape_ops;
#[path = "linalg/solve.rs"]
mod solve_ops;
#[path = "indexing/sort.rs"]
mod sort_ops;
#[path = "manipulation/split.rs"]
mod split_ops;
#[path = "manipulation/stack.rs"]
mod stack_ops;
#[path = "indexing/take.rs"]
mod take_ops;
#[cfg(feature = "test-support")]
#[path = "support/test_support.rs"]
mod test_support;
#[path = "manipulation/tile.rs"]
mod tile_ops;
#[path = "linalg/trace.rs"]
mod trace_ops;
#[path = "operations/unary.rs"]
mod unary;
#[path = "indexing/unique.rs"]
mod unique_ops;
#[path = "operations/where_ops.rs"]
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
    module.add_function(wrap_pyfunction!(eye, module)?)?;
    module.add_function(wrap_pyfunction!(identity, module)?)?;
    module.add_function(wrap_pyfunction!(full, module)?)?;
    module.add_function(wrap_pyfunction!(arange, module)?)?;
    module.add_function(wrap_pyfunction!(linspace, module)?)?;
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
    module.add_function(wrap_pyfunction!(bitwise_and, module)?)?;
    module.add_function(wrap_pyfunction!(bitwise_or, module)?)?;
    module.add_function(wrap_pyfunction!(bitwise_xor, module)?)?;
    module.add_function(wrap_pyfunction!(bitwise_not, module)?)?;
    module.add_function(wrap_pyfunction!(take, module)?)?;
    module.add_function(wrap_pyfunction!(dot, module)?)?;
    module.add_function(wrap_pyfunction!(norm, module)?)?;
    module.add_function(wrap_pyfunction!(trace, module)?)?;
    module.add_function(wrap_pyfunction!(concatenate, module)?)?;
    module.add_function(wrap_pyfunction!(stack, module)?)?;
    module.add_function(wrap_pyfunction!(diag, module)?)?;
    module.add_function(wrap_pyfunction!(matrix_norm, module)?)?;
    module.add_function(wrap_pyfunction!(det, module)?)?;
    module.add_function(wrap_pyfunction!(inverse, module)?)?;
    module.add_function(wrap_pyfunction!(solve, module)?)?;
    module.add_function(wrap_pyfunction!(ravel, module)?)?;
    module.add_function(wrap_pyfunction!(flatten, module)?)?;
    module.add_function(wrap_pyfunction!(nonzero, module)?)?;
    module.add_function(wrap_pyfunction!(argwhere, module)?)?;
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
    module.add_function(wrap_pyfunction!(squeeze, module)?)?;
    module.add_function(wrap_pyfunction!(expand_dims, module)?)?;
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
fn asarray(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    constructors::asarray(py, value, dtype)
}

#[pyfunction(signature = (shape, dtype = None))]
fn zeros(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    constructors::zeros(py, shape, dtype)
}

#[pyfunction(signature = (shape, dtype = None))]
fn ones(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    constructors::ones(py, shape, dtype)
}

#[pyfunction(signature = (rows, columns = None, dtype = None))]
fn eye(
    py: Python<'_>,
    rows: usize,
    columns: Option<usize>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    constructors::eye(py, rows, columns, dtype)
}

#[pyfunction(signature = (size, dtype = None))]
fn identity(py: Python<'_>, size: usize, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    constructors::identity(py, size, dtype)
}

#[pyfunction(signature = (shape, fill_value, dtype = None))]
fn full(
    py: Python<'_>,
    shape: &Bound<'_, PyAny>,
    fill_value: &Bound<'_, PyAny>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    constructors::full(py, shape, fill_value, dtype)
}

#[pyfunction(signature = (start, stop = None, step = None, dtype = None))]
fn arange(
    py: Python<'_>,
    start: &Bound<'_, PyAny>,
    stop: Option<&Bound<'_, PyAny>>,
    step: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    constructors::arange(py, start, stop, step, dtype)
}

#[pyfunction(signature = (start, stop, num, dtype = None, endpoint = true))]
fn linspace(
    py: Python<'_>,
    start: f64,
    stop: f64,
    num: usize,
    dtype: Option<&Bound<'_, PyAny>>,
    endpoint: bool,
) -> PyResult<Py<PyAny>> {
    constructors::linspace(py, start, stop, num, dtype, endpoint)
}

#[pyfunction(signature = (value, dtype, copy = true))]
fn astype(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    dtype: &Bound<'_, PyAny>,
    copy: bool,
) -> PyResult<Py<PyAny>> {
    casting::astype(py, value, dtype, copy)
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

#[pyfunction(signature = (value, indices, axis = None))]
fn take(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    indices: Vec<i64>,
    axis: Option<i64>,
) -> PyResult<Py<PyAny>> {
    take_ops::take(py, value, indices, axis)
}

#[pyfunction]
fn dot(py: Python<'_>, lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    dot_ops::dot(py, lhs, rhs)
}

#[pyfunction]
fn norm(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<f64> {
    norm_ops::norm(py, value)
}

#[pyfunction]
fn trace(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    trace_ops::trace(py, value)
}

#[pyfunction(signature = (arrays, axis = 0))]
fn concatenate(py: Python<'_>, arrays: Vec<Py<PyAny>>, axis: i64) -> PyResult<Py<PyAny>> {
    concat_ops::concatenate(py, arrays, axis)
}

#[pyfunction(signature = (arrays, axis = 0))]
fn stack(py: Python<'_>, arrays: Vec<Py<PyAny>>, axis: i64) -> PyResult<Py<PyAny>> {
    stack_ops::stack(py, arrays, axis)
}

#[pyfunction(signature = (value, k = 0))]
fn diag(py: Python<'_>, value: &Bound<'_, PyAny>, k: isize) -> PyResult<Py<PyAny>> {
    diag_ops::diag(py, value, k)
}

#[pyfunction(signature = (value, order = "fro"))]
fn matrix_norm(py: Python<'_>, value: &Bound<'_, PyAny>, order: &str) -> PyResult<f64> {
    matrix_norm_ops::matrix_norm(py, value, order)
}

#[pyfunction]
fn det(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<f64> {
    determinant_ops::det(py, value)
}

#[pyfunction]
fn inverse(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    inverse_ops::inverse(py, value)
}

#[pyfunction]
fn solve(py: Python<'_>, matrix: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    solve_ops::solve(py, matrix, rhs)
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

#[pyfunction]
fn variance(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    reduction::variance(py, value)
}

#[pyfunction]
fn stddev(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    reduction::stddev(py, value)
}

#[pyfunction(signature = (value, axis = None))]
fn argmin(py: Python<'_>, value: &Bound<'_, PyAny>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
    reduction::argmin(py, value, axis)
}

#[pyfunction(signature = (value, axis = None))]
fn argmax(py: Python<'_>, value: &Bound<'_, PyAny>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
    reduction::argmax(py, value, axis)
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

#[pyfunction]
fn unique(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    unique_ops::unique(py, value)
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

#[pyfunction(signature = (sorted, values, side = "left"))]
fn searchsorted(
    py: Python<'_>,
    sorted: &Bound<'_, PyAny>,
    values: &Bound<'_, PyAny>,
    side: &str,
) -> PyResult<Py<PyAny>> {
    searchsorted_ops::searchsorted(py, sorted, values, side)
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

#[pyfunction(signature = (value, axis = None))]
fn squeeze(py: Python<'_>, value: &Bound<'_, PyAny>, axis: Option<i64>) -> PyResult<Py<PyAny>> {
    shape_ops::squeeze(py, value, axis)
}

#[pyfunction]
fn expand_dims(py: Python<'_>, value: &Bound<'_, PyAny>, axis: i64) -> PyResult<Py<PyAny>> {
    shape_ops::expand_dims(py, value, axis)
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
