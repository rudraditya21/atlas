use std::fmt::Display;

use atlas_linalg::AtlasLinalgError;
use atlas_ndarray::AtlasNdError;
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyModule};
#[cfg(feature = "test-support")]
use {atlas_ml::AtlasMlError, atlas_random::AtlasRandomError, atlas_stats::AtlasStatsError};

pub(crate) fn ndarray(py: Python<'_>, error: AtlasNdError) -> PyErr {
    let exception = match &error {
        AtlasNdError::InvalidAxis { .. } | AtlasNdError::IndexOutOfBounds { .. } => "AxisError",
        AtlasNdError::InvalidSlice { .. } => "SliceError",
        AtlasNdError::ShapeMismatch { .. }
        | AtlasNdError::ShapeOverflow { .. }
        | AtlasNdError::DimensionMismatch { .. }
        | AtlasNdError::InvalidReshape { .. }
        | AtlasNdError::InvalidBroadcast { .. }
        | AtlasNdError::MaskShapeMismatch { .. }
        | AtlasNdError::InvalidArgument { op: "permute_axes", .. }
        | AtlasNdError::InvalidArgument { op: "moveaxis", .. }
        | AtlasNdError::InvalidArgument { op: "reshape", .. }
        | AtlasNdError::InvalidArgument { op: "squeeze", .. }
        | AtlasNdError::InvalidArgument { op: "concatenate", .. }
        | AtlasNdError::InvalidArgument { op: "stack", .. }
        | AtlasNdError::InvalidShape => "ShapeError",
        AtlasNdError::EmptyReduction { .. }
        | AtlasNdError::AllNaN { .. }
        | AtlasNdError::NumericConversionFailed { .. }
        | AtlasNdError::DivisionByZero { .. }
        | AtlasNdError::InvalidArgument { .. }
        | AtlasNdError::InvalidCast { .. } => "NumericError",
    };

    python_error(py, exception, error)
}

#[cfg(feature = "test-support")]
pub(crate) fn stats(py: Python<'_>, error: AtlasStatsError) -> PyErr {
    match &error {
        AtlasStatsError::NdArray(error) => ndarray(py, error.clone()),
        AtlasStatsError::InvalidInputRank { .. } | AtlasStatsError::ShapeMismatch { .. } => {
            python_error(py, "ShapeError", error)
        }
        AtlasStatsError::EmptyInput { .. }
        | AtlasStatsError::InvalidDegreesOfFreedom { .. }
        | AtlasStatsError::NumericConversionFailed { .. }
        | AtlasStatsError::InvalidWeights { .. }
        | AtlasStatsError::InvalidQuantile { .. }
        | AtlasStatsError::ZeroVariance { .. } => python_error(py, "NumericError", error),
    }
}

pub(crate) fn linalg(py: Python<'_>, error: AtlasLinalgError) -> PyErr {
    match &error {
        AtlasLinalgError::NdArray(error) => ndarray(py, error.clone()),
        AtlasLinalgError::InvalidOperandRank { .. }
        | AtlasLinalgError::ShapeMismatch { .. }
        | AtlasLinalgError::InvalidInputRank { .. }
        | AtlasLinalgError::InvalidInputShape { .. }
        | AtlasLinalgError::InvalidFactor { .. } => python_error(py, "ShapeError", error),
        AtlasLinalgError::NonFiniteInput { .. }
        | AtlasLinalgError::SingularMatrix { .. }
        | AtlasLinalgError::RankDeficientMatrix { .. }
        | AtlasLinalgError::NotPositiveDefinite { .. }
        | AtlasLinalgError::InvalidArgument { .. }
        | AtlasLinalgError::IterationLimit { .. } => python_error(py, "NumericError", error),
    }
}

#[cfg(feature = "test-support")]
pub(crate) fn random(py: Python<'_>, error: AtlasRandomError) -> PyErr {
    match &error {
        AtlasRandomError::NdArray(error) => ndarray(py, error.clone()),
        AtlasRandomError::InvalidArgument { .. }
        | AtlasRandomError::DistributionInitializationFailed { .. } => {
            python_error(py, "NumericError", error)
        }
    }
}

#[cfg(feature = "test-support")]
pub(crate) fn ml(py: Python<'_>, error: AtlasMlError) -> PyErr {
    match &error {
        AtlasMlError::NdArray(error) => ndarray(py, error.clone()),
        AtlasMlError::Linalg(error) => linalg(py, error.clone()),
        error => python_error(py, "ModelError", error),
    }
}

fn python_error(py: Python<'_>, exception: &str, error: impl Display) -> PyErr {
    let message = error.to_string();
    let error_instance = PyModule::import(py, "atlas.errors")
        .and_then(|module| module.getattr(exception))
        .and_then(|error_type| error_type.call1((message.clone(),)));

    match error_instance {
        Ok(error_instance) => PyErr::from_value(error_instance),
        Err(_) => PyRuntimeError::new_err(message),
    }
}
