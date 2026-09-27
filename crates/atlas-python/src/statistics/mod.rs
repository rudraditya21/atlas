//! Statistics Python binding modules.

#[path = "kurtosis.rs"]
mod kurtosis_ops;
#[path = "pairwise.rs"]
mod pairwise_ops;
#[path = "quantile.rs"]
mod quantile_ops;
#[path = "skewness.rs"]
mod skewness_ops;
#[path = "weighted.rs"]
mod weighted_ops;

use pyo3::{prelude::*, wrap_pyfunction};

use crate::reduction;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(variance, module)?)?;
    module.add_function(wrap_pyfunction!(stddev, module)?)?;
    module.add_function(wrap_pyfunction!(kurtosis, module)?)?;
    module.add_function(wrap_pyfunction!(skewness, module)?)?;
    module.add_function(wrap_pyfunction!(median, module)?)?;
    module.add_function(wrap_pyfunction!(quantile, module)?)?;
    module.add_function(wrap_pyfunction!(median_axis, module)?)?;
    module.add_function(wrap_pyfunction!(quantile_axis, module)?)?;
    module.add_function(wrap_pyfunction!(weighted_mean, module)?)?;
    module.add_function(wrap_pyfunction!(weighted_variance, module)?)?;
    module.add_function(wrap_pyfunction!(covariance, module)?)?;
    module.add_function(wrap_pyfunction!(correlation, module)?)?;
    module.add_function(wrap_pyfunction!(covariance_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(correlation_matrix, module)?)?;
    Ok(())
}

#[pyfunction(signature = (value, ddof = 0))]
fn variance(py: Python<'_>, value: &Bound<'_, PyAny>, ddof: usize) -> PyResult<Py<PyAny>> {
    reduction::variance(py, value, ddof)
}

#[pyfunction(signature = (value, ddof = 0))]
fn stddev(py: Python<'_>, value: &Bound<'_, PyAny>, ddof: usize) -> PyResult<Py<PyAny>> {
    reduction::stddev(py, value, ddof)
}

#[pyfunction]
fn kurtosis(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<f64> {
    kurtosis_ops::kurtosis(py, value)
}

#[pyfunction]
fn skewness(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<f64> {
    skewness_ops::skewness(py, value)
}

#[pyfunction]
fn median(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<f64> {
    quantile_ops::median(py, value)
}

#[pyfunction(signature = (value, q, interpolation = "linear"))]
fn quantile(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    q: f64,
    interpolation: &str,
) -> PyResult<f64> {
    quantile_ops::quantile(py, value, q, interpolation)
}

#[pyfunction]
fn median_axis(py: Python<'_>, value: &Bound<'_, PyAny>, axis: i64) -> PyResult<Py<PyAny>> {
    quantile_ops::median_axis(py, value, axis)
}

#[pyfunction(signature = (value, q, axis, interpolation = "linear"))]
fn quantile_axis(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    q: f64,
    axis: i64,
    interpolation: &str,
) -> PyResult<Py<PyAny>> {
    quantile_ops::quantile_axis(py, value, q, axis, interpolation)
}

#[pyfunction]
fn weighted_mean(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
    weights: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    weighted_ops::weighted_mean(py, values, weights)
}

#[pyfunction]
fn weighted_variance(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
    weights: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    weighted_ops::weighted_variance(py, values, weights)
}

#[pyfunction]
fn covariance(py: Python<'_>, lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<f64> {
    pairwise_ops::covariance(py, lhs, rhs)
}

#[pyfunction]
fn correlation(py: Python<'_>, lhs: &Bound<'_, PyAny>, rhs: &Bound<'_, PyAny>) -> PyResult<f64> {
    pairwise_ops::correlation(py, lhs, rhs)
}

#[pyfunction]
fn covariance_matrix(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    pairwise_ops::covariance_matrix(py, value)
}

#[pyfunction]
fn correlation_matrix(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    pairwise_ops::correlation_matrix(py, value)
}
