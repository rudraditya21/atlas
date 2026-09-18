use atlas_linalg::AtlasLinalgError;
use atlas_ml::AtlasMlError;
use atlas_ndarray::AtlasNdError;
use atlas_random::AtlasRandomError;
use atlas_stats::AtlasStatsError;
use pyo3::prelude::*;

use crate::{error, scalar};

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(raise_ndarray_error, module)?)?;
    module.add_function(wrap_pyfunction!(raise_stats_error, module)?)?;
    module.add_function(wrap_pyfunction!(raise_linalg_error, module)?)?;
    module.add_function(wrap_pyfunction!(raise_random_error, module)?)?;
    module.add_function(wrap_pyfunction!(raise_ml_error, module)?)?;
    module.add_function(wrap_pyfunction!(scalar_kind, module)?)
}

#[pyfunction(name = "_raise_ndarray_error")]
fn raise_ndarray_error(py: Python<'_>) -> PyResult<()> {
    Err(error::ndarray(py, AtlasNdError::InvalidAxis { axis: 1, ndim: 1 }))
}

#[pyfunction(name = "_raise_stats_error")]
fn raise_stats_error(py: Python<'_>) -> PyResult<()> {
    Err(error::stats(py, AtlasStatsError::InvalidQuantile { reason: "must be within [0, 1]" }))
}

#[pyfunction(name = "_raise_linalg_error")]
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

#[pyfunction(name = "_raise_random_error")]
fn raise_random_error(py: Python<'_>) -> PyResult<()> {
    Err(error::random(
        py,
        AtlasRandomError::InvalidArgument { op: "uniform", reason: "low must be less than high" },
    ))
}

#[pyfunction(name = "_raise_ml_error")]
fn raise_ml_error(py: Python<'_>) -> PyResult<()> {
    Err(error::ml(
        py,
        AtlasMlError::InvalidArgument { op: "knn_fit", reason: "k must be positive" },
    ))
}

#[pyfunction(name = "_scalar_kind")]
fn scalar_kind(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<&'static str> {
    scalar::from_python(py, value).map(scalar::kind)
}
