use pyo3::prelude::*;

use crate::support::{arrays as array, gil};

const FIT_OP: &str = "decision_stump_fit";
const PREDICT_OP: &str = "decision_stump_predict";

#[pyclass(module = "atlas._native")]
pub(crate) struct DecisionStumpClassifier {
    model: super::model::NativeModel<atlas_ml::DecisionStumpClassifier>,
}

#[pymethods]
impl DecisionStumpClassifier {
    #[new]
    fn new() -> Self {
        Self { model: super::model::NativeModel::new() }
    }

    fn __repr__(&self) -> &'static str {
        "DecisionStumpClassifier()"
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.model.is_fitted()
    }

    #[getter]
    fn feature_index_(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.model.fitted(py, PREDICT_OP)?.feature_index())
    }

    #[getter]
    fn threshold_(&self, py: Python<'_>) -> PyResult<f64> {
        Ok(self.model.fitted(py, PREDICT_OP)?.threshold())
    }

    #[getter]
    fn left_label_(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.model.fitted(py, PREDICT_OP)?.left_label())
    }

    #[getter]
    fn right_label_(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.model.fitted(py, PREDICT_OP)?.right_label())
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        labels: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let model = super::model::with_classifier_fit_inputs(
            py,
            features,
            labels,
            FIT_OP,
            |features, labels| {
                gil::without_gil(py, || atlas_ml::DecisionStumpClassifier::fit(features, labels))
            },
        )?;

        slf.model.replace(model);
        Ok(slf)
    }

    fn predict(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
    ) -> crate::support::results::PyObjectResult {
        let model = self.model.fitted(py, PREDICT_OP)?;
        let predictions =
            super::model::with_predict_features(py, features, PREDICT_OP, |features| {
                gil::without_gil(py, || model.predict(features))
            })?;

        Ok(array::to_numpy_owned(py, predictions)?.into_any().unbind())
    }

    fn fit_predict(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        labels: &Bound<'_, PyAny>,
    ) -> crate::support::results::PyObjectResult {
        let (model, predictions) = super::model::with_classifier_fit_inputs(
            py,
            features,
            labels,
            FIT_OP,
            |features, labels| {
                let model = gil::without_gil(py, || {
                    atlas_ml::DecisionStumpClassifier::fit(features, labels)
                })?;
                let predictions = gil::without_gil(py, || model.predict(features))?;
                Ok((model, predictions))
            },
        )?;
        slf.model.replace(model);
        Ok(array::to_numpy_owned(py, predictions)?.into_any().unbind())
    }
}
