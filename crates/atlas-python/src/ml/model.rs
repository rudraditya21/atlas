use atlas_ml::AtlasMlError;
use atlas_ndarray::NDArray;
use pyo3::prelude::*;

use crate::array;

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

    pub(crate) fn fitted(&self, py: Python<'_>, operation: &'static str) -> PyResult<&T> {
        self.inner.as_ref().ok_or_else(|| {
            crate::error::ml(
                py,
                AtlasMlError::InvalidArgument {
                    op: operation,
                    reason: "model must be fitted before prediction",
                },
            )
        })
    }
}

pub(crate) fn classifier_fit_inputs(
    py: Python<'_>,
    features: &Bound<'_, PyAny>,
    labels: &Bound<'_, PyAny>,
    operation: &'static str,
) -> PyResult<(NDArray<f64>, NDArray<usize>)> {
    let features = array::feature_matrix_f64(py, features)?;
    let labels = array::label_vector_usize(py, labels)?;
    validate_fit_inputs(&features, &labels, operation)
        .map_err(|error| crate::error::ml(py, error))?;

    Ok((features, labels))
}

pub(crate) fn predict_features(
    py: Python<'_>,
    features: &Bound<'_, PyAny>,
    operation: &'static str,
) -> PyResult<NDArray<f64>> {
    let features = array::feature_matrix_f64(py, features)?;
    if features.ndim() != 2 {
        return Err(crate::error::ml(
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

fn validate_fit_inputs(
    features: &NDArray<f64>,
    labels: &NDArray<usize>,
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
            expected: "a rank-1 label vector",
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
