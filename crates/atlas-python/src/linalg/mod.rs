//! Linear algebra Python binding modules.

#[path = "cholesky.rs"]
mod cholesky_ops;
#[path = "conjugate_gradient.rs"]
mod conjugate_gradient_ops;
#[path = "determinant.rs"]
mod determinant_ops;
#[path = "diag.rs"]
mod diag_ops;
#[path = "dot.rs"]
mod dot_ops;
#[path = "eigen.rs"]
mod eigen_ops;
#[path = "inverse.rs"]
mod inverse_ops;
#[path = "matmul.rs"]
mod matmul_ops;
#[path = "matrix_norm.rs"]
mod matrix_norm_ops;
#[path = "norm.rs"]
mod norm_ops;
#[path = "qr.rs"]
mod qr_ops;
#[path = "slogdet.rs"]
mod slogdet_ops;
#[path = "solve.rs"]
mod solve_ops;
#[path = "trace.rs"]
mod trace_ops;

use pyo3::{prelude::*, wrap_pyfunction};

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<conjugate_gradient_ops::ConjugateGradientResult>()?;
    module.add_function(wrap_pyfunction!(dot, module)?)?;
    module.add_function(wrap_pyfunction!(norm, module)?)?;
    module.add_function(wrap_pyfunction!(trace, module)?)?;
    module.add_function(wrap_pyfunction!(diag, module)?)?;
    module.add_function(wrap_pyfunction!(matrix_norm, module)?)?;
    module.add_function(wrap_pyfunction!(det, module)?)?;
    module.add_function(wrap_pyfunction!(inverse, module)?)?;
    module.add_function(wrap_pyfunction!(solve, module)?)?;
    module.add_function(wrap_pyfunction!(solve_transpose, module)?)?;
    module.add_function(wrap_pyfunction!(slogdet, module)?)?;
    module.add_function(wrap_pyfunction!(cholesky, module)?)?;
    module.add_function(wrap_pyfunction!(solve_spd, module)?)?;
    module.add_function(wrap_pyfunction!(qr, module)?)?;
    module.add_function(wrap_pyfunction!(least_squares, module)?)?;
    module.add_function(wrap_pyfunction!(matrix_rank, module)?)?;
    module.add_function(wrap_pyfunction!(symmetric_eigendecomposition, module)?)?;
    module.add_function(wrap_pyfunction!(conjugate_gradient, module)?)?;
    module.add_function(wrap_pyfunction!(matmul, module)?)?;
    Ok(())
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
fn solve_transpose(
    py: Python<'_>,
    matrix: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    solve_ops::solve_transpose(py, matrix, rhs)
}

#[pyfunction]
fn slogdet(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<(f64, f64)> {
    slogdet_ops::slogdet(py, value)
}

#[pyfunction]
fn cholesky(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    cholesky_ops::cholesky(py, value)
}

#[pyfunction]
fn solve_spd(
    py: Python<'_>,
    matrix: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    cholesky_ops::solve_spd(py, matrix, rhs)
}

#[pyfunction]
fn qr(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    qr_ops::qr(py, value)
}

#[pyfunction]
fn least_squares(
    py: Python<'_>,
    matrix: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    qr_ops::least_squares(py, matrix, rhs)
}

#[pyfunction]
fn matrix_rank(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<usize> {
    qr_ops::matrix_rank(py, value)
}

#[pyfunction]
fn symmetric_eigendecomposition(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    eigen_ops::symmetric_eigendecomposition(py, value)
}

#[pyfunction]
fn conjugate_gradient(
    py: Python<'_>,
    matrix: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    max_iterations: usize,
    tolerance: f64,
) -> PyResult<Py<conjugate_gradient_ops::ConjugateGradientResult>> {
    conjugate_gradient_ops::conjugate_gradient(py, matrix, rhs, max_iterations, tolerance)
}

#[pyfunction]
fn matmul(py: Python<'_>, lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    matmul_ops::matmul(py, lhs, rhs)
}
