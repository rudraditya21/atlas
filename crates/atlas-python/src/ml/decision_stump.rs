use pyo3::prelude::*;

use crate::{array, gil};

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
        .map_err(|error| crate::error::ml(py, error))?;

        slf.model.replace(model);
        Ok(slf)
    }

    fn predict(&self, py: Python<'_>, features: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let features = super::model::predict_features(py, features, PREDICT_OP)?;
        let model = self.model.fitted(py, PREDICT_OP)?;
        let predictions = gil::without_gil(py, move || model.predict(&features))
            .map_err(|error| crate::error::ml(py, error))?;

        Ok(array::to_numpy_owned(py, predictions)?.into_any().unbind())
    }
}
