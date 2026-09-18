use atlas_linalg::AtlasLinalgError;
use atlas_ml::AtlasMlError;
use atlas_ndarray::AtlasNdError;
use atlas_random::AtlasRandomError;
use atlas_stats::AtlasStatsError;
use pyo3::prelude::*;

use crate::error;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(raise_ndarray_error, module)?)?;
    module.add_function(wrap_pyfunction!(raise_stats_error, module)?)?;
    module.add_function(wrap_pyfunction!(raise_linalg_error, module)?)?;
    module.add_function(wrap_pyfunction!(raise_random_error, module)?)?;
    module.add_function(wrap_pyfunction!(raise_ml_error, module)?)
}

#[pyfunction]
fn raise_ndarray_error(py: Python<'_>) -> PyResult<()> {
    Err(error::ndarray(py, AtlasNdError::InvalidAxis { axis: 1, ndim: 1 }))
}

#[pyfunction]
fn raise_stats_error(py: Python<'_>) -> PyResult<()> {
    Err(error::stats(py, AtlasStatsError::InvalidQuantile { reason: "must be within [0, 1]" }))
}

#[pyfunction]
fn raise_linalg_error(py: Python<'_>) -> PyResult<()> {
    Err(error::linalg(
        py,
        AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: vec![2, 3],
            right: vec![4, 2],
            reason: "inner dimensions must match",
        },
    ))
}

#[pyfunction]
fn raise_random_error(py: Python<'_>) -> PyResult<()> {
    Err(error::random(
        py,
        AtlasRandomError::InvalidArgument { op: "uniform", reason: "low must be less than high" },
    ))
}

#[pyfunction]
fn raise_ml_error(py: Python<'_>) -> PyResult<()> {
    Err(error::ml(
        py,
        AtlasMlError::InvalidArgument { op: "knn_fit", reason: "k must be positive" },
    ))
}
