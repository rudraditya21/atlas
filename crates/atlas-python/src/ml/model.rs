use atlas_ml::AtlasMlError;
use atlas_ndarray::{ArrayElement, NDArray};
use pyo3::prelude::*;

use crate::support::arrays as array;

pub(crate) struct NativeModel<T> {
    inner: Option<T>,
}

impl<T> NativeModel<T> {
    pub(crate) const fn new() -> Self {
        Self { inner: None }
    }

    pub(crate) fn replace(&mut self, model: T) {
        self.inner = Some(model);
    }

    pub(crate) fn clear(&mut self) {
        self.inner = None;
    }

    pub(crate) const fn is_fitted(&self) -> bool {
        self.inner.is_some()
    }

    pub(crate) fn fitted(&self, py: Python<'_>, operation: &'static str) -> PyResult<&T> {
        self.inner.as_ref().ok_or_else(|| {
            crate::support::errors::ml(
                py,
                AtlasMlError::InvalidArgument {
                    op: operation,
                    reason: "model must be fitted before prediction",
                },
            )
        })
    }
}

pub(crate) fn unexpected_parameter(name: &str) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!("unknown parameter: {name}"))
}

pub(crate) fn classifier_fit_inputs(
    py: Python<'_>,
    features: &Bound<'_, PyAny>,
    labels: &Bound<'_, PyAny>,
    operation: &'static str,
) -> PyResult<(NDArray<f64>, NDArray<usize>)> {
    let features = array::feature_matrix_f64(py, features)?;
    let labels = array::label_vector_usize(py, labels)?;
    validate_fit_inputs(&features, &labels, "a rank-1 label vector", operation)
        .map_err(|error| crate::support::errors::ml(py, error))?;

    Ok((features, labels))
}

pub(crate) fn regression_fit_inputs(
    py: Python<'_>,
    features: &Bound<'_, PyAny>,
    targets: &Bound<'_, PyAny>,
    operation: &'static str,
) -> PyResult<(NDArray<f64>, NDArray<f64>)> {
    let features = array::feature_matrix_f64(py, features)?;
    let targets = array::target_vector_f64(py, targets)?;
    validate_fit_inputs(&features, &targets, "a rank-1 target vector", operation)
        .map_err(|error| crate::support::errors::ml(py, error))?;

    Ok((features, targets))
}

pub(crate) fn predict_features(
    py: Python<'_>,
    features: &Bound<'_, PyAny>,
    operation: &'static str,
) -> PyResult<NDArray<f64>> {
    let features = array::feature_matrix_f64(py, features)?;
    if features.ndim() != 2 {
        return Err(crate::support::errors::ml(
            py,
            AtlasMlError::InvalidInputRank {
                op: operation,
                expected: "a rank-2 [queries, features] matrix",
                rank: features.ndim(),
            },
        ));
    }

    Ok(features)
}

fn validate_fit_inputs<T: ArrayElement>(
    features: &NDArray<f64>,
    labels: &NDArray<T>,
    target_description: &'static str,
    operation: &'static str,
) -> atlas_ml::AtlasMlResult<()> {
    if features.ndim() != 2 {
        return Err(AtlasMlError::InvalidInputRank {
            op: operation,
            expected: "a rank-2 [samples, features] matrix",
            rank: features.ndim(),
        });
    }
    if labels.ndim() != 1 {
        return Err(AtlasMlError::InvalidInputRank {
            op: operation,
            expected: target_description,
            rank: labels.ndim(),
        });
    }
    if features.shape()[0] == 0 || labels.shape()[0] == 0 {
        return Err(AtlasMlError::EmptyInput { op: operation });
    }
    if features.shape()[0] != labels.shape()[0] {
        return Err(AtlasMlError::ShapeMismatch {
            op: operation,
            left: features.shape().to_vec(),
            right: labels.shape().to_vec(),
            reason: "sample counts must match",
        });
    }

    Ok(())
}
