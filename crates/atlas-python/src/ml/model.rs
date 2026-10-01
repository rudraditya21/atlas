use atlas_ml::AtlasMlError;
use atlas_ndarray::{ArrayElement, NDArray, OperandMetadata};
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

pub(crate) fn owned_classifier_fit_inputs(
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

pub(crate) fn owned_regression_fit_inputs(
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

pub(crate) fn with_classifier_fit_inputs<R>(
    py: Python<'_>,
    features: &Bound<'_, PyAny>,
    labels: &Bound<'_, PyAny>,
    operation: &'static str,
    fit: impl FnOnce(
        &(dyn OperandMetadata<f64> + Sync),
        &(dyn OperandMetadata<usize> + Sync),
    ) -> atlas_ml::AtlasMlResult<R>,
) -> PyResult<R> {
    let features = array::readonly_from_python::<f64>(py, features)?;
    let labels = array::readonly_from_python::<usize>(py, labels)?;
    let result = array::with_numpy_operand(features, |features| {
        array::with_numpy_operand(labels, |labels| {
            validate_fit_inputs(features, labels, "a rank-1 label vector", operation)?;
            fit(features, labels)
        })
    })??;

    result.map_err(|error| crate::support::errors::ml(py, error))
}

pub(crate) fn with_regression_fit_inputs<R>(
    py: Python<'_>,
    features: &Bound<'_, PyAny>,
    targets: &Bound<'_, PyAny>,
    operation: &'static str,
    fit: impl FnOnce(
        &(dyn OperandMetadata<f64> + Sync),
        &(dyn OperandMetadata<f64> + Sync),
    ) -> atlas_ml::AtlasMlResult<R>,
) -> PyResult<R> {
    let features = array::readonly_from_python::<f64>(py, features)?;
    let targets = array::readonly_from_python::<f64>(py, targets)?;
    let result = array::with_numpy_operand(features, |features| {
        array::with_numpy_operand(targets, |targets| {
            validate_fit_inputs(features, targets, "a rank-1 target vector", operation)?;
            fit(features, targets)
        })
    })??;

    result.map_err(|error| crate::support::errors::ml(py, error))
}

pub(crate) fn with_predict_features<R>(
    py: Python<'_>,
    features: &Bound<'_, PyAny>,
    operation: &'static str,
    predict: impl FnOnce(&(dyn OperandMetadata<f64> + Sync)) -> atlas_ml::AtlasMlResult<R>,
) -> PyResult<R> {
    let features = array::readonly_from_python::<f64>(py, features)?;
    let result = array::with_numpy_operand(features, |features| {
        if features.ndim() != 2 {
            return Err(AtlasMlError::InvalidInputRank {
                op: operation,
                expected: "a rank-2 [queries, features] matrix",
                rank: features.ndim(),
            });
        }

        predict(features)
    })?;

    result.map_err(|error| crate::support::errors::ml(py, error))
}

fn validate_fit_inputs<T, F, L>(
    features: &F,
    labels: &L,
    target_description: &'static str,
    operation: &'static str,
) -> atlas_ml::AtlasMlResult<()>
where
    T: ArrayElement,
    F: OperandMetadata<f64> + ?Sized,
    L: OperandMetadata<T> + ?Sized,
{
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
