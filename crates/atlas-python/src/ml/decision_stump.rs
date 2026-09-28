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
        let (features, labels) = super::model::classifier_fit_inputs(py, features, labels, FIT_OP)?;
        let model = gil::without_gil(py, move || {
            atlas_ml::DecisionStumpClassifier::fit(&features, &labels)
        })
        .map_err(|error| crate::support::errors::ml(py, error))?;

        slf.model.replace(model);
        Ok(slf)
    }

    fn predict(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
    ) -> crate::support::results::PyObjectResult {
        let features = super::model::predict_features(py, features, PREDICT_OP)?;
        let model = self.model.fitted(py, PREDICT_OP)?;
        let predictions = gil::without_gil(py, move || model.predict(&features))
            .map_err(|error| crate::support::errors::ml(py, error))?;

        Ok(array::to_numpy_owned(py, predictions)?.into_any().unbind())
    }

    fn fit_predict(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        labels: &Bound<'_, PyAny>,
    ) -> crate::support::results::PyObjectResult {
        let (features, labels) = super::model::classifier_fit_inputs(py, features, labels, FIT_OP)?;
        let model =
            gil::without_gil(py, || atlas_ml::DecisionStumpClassifier::fit(&features, &labels))
                .map_err(|error| crate::support::errors::ml(py, error))?;
        let predictions = gil::without_gil(py, || model.predict(&features))
            .map_err(|error| crate::support::errors::ml(py, error))?;
        slf.model.replace(model);
        Ok(array::to_numpy_owned(py, predictions)?.into_any().unbind())
    }
}
