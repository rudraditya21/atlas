use pyo3::prelude::*;

use crate::support::{arrays as array, gil};

const FIT_OP: &str = "nearest_centroid_fit";
const PREDICT_OP: &str = "nearest_centroid_predict";

#[pyclass(module = "atlas._native")]
pub(crate) struct NearestCentroidClassifier {
    model: super::model::NativeModel<atlas_ml::NearestCentroidClassifier>,
}

#[pymethods]
impl NearestCentroidClassifier {
    #[new]
    fn new() -> Self {
        Self { model: super::model::NativeModel::new() }
    }

    fn __repr__(&self) -> &'static str {
        "NearestCentroidClassifier()"
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.model.is_fitted()
    }

    #[getter]
    fn classes_(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(array::to_numpy_usize_vector(py, self.model.fitted(py, PREDICT_OP)?.classes()))
    }

    #[getter]
    fn centroids_(&self, py: Python<'_>) -> crate::support::results::PyObjectResult {
        let centroids = self.model.fitted(py, PREDICT_OP)?.centroids().clone();
        Ok(array::to_numpy_owned(py, centroids)?.into_any().unbind())
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
                gil::without_gil(py, || atlas_ml::NearestCentroidClassifier::fit(features, labels))
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
                    atlas_ml::NearestCentroidClassifier::fit(features, labels)
                })?;
                let predictions = gil::without_gil(py, || model.predict(features))?;
                Ok((model, predictions))
            },
        )?;
        slf.model.replace(model);
        Ok(array::to_numpy_owned(py, predictions)?.into_any().unbind())
    }
}
