use pyo3::prelude::*;

use crate::support::{arrays as array, gil};

const FIT_OP: &str = "gaussian_naive_bayes_fit";
const PREDICT_PROBA_OP: &str = "gaussian_naive_bayes_predict_proba";
const PREDICT_OP: &str = "gaussian_naive_bayes_predict";

#[pyclass(module = "atlas._native")]
pub(crate) struct GaussianNaiveBayes {
    config: atlas_ml::GaussianNaiveBayesConfig,
    model: super::model::NativeModel<atlas_ml::GaussianNaiveBayes>,
}

#[pymethods]
impl GaussianNaiveBayes {
    #[new]
    #[pyo3(signature = (variance_smoothing = 1e-9))]
    fn new(py: Python<'_>, variance_smoothing: f64) -> PyResult<Self> {
        let config = atlas_ml::GaussianNaiveBayesConfig::new(variance_smoothing)
            .map_err(|error| crate::support::errors::ml(py, error))?;

        Ok(Self { config, model: super::model::NativeModel::new() })
    }

    fn __repr__(&self) -> String {
        format!("GaussianNaiveBayes(variance_smoothing={})", self.config.variance_smoothing())
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.model.is_fitted()
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        labels: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (features, labels) = super::model::classifier_fit_inputs(py, features, labels, FIT_OP)?;
        let config = slf.config;
        let model = gil::without_gil(py, move || {
            atlas_ml::GaussianNaiveBayes::fit(&features, &labels, config)
        })
        .map_err(|error| crate::support::errors::ml(py, error))?;

        slf.model.replace(model);
        Ok(slf)
    }

    fn predict_proba(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
    ) -> crate::support::results::PyObjectResult {
        let features = super::model::predict_features(py, features, PREDICT_PROBA_OP)?;
        let model = self.model.fitted(py, PREDICT_PROBA_OP)?;
        let probabilities = gil::without_gil(py, move || model.predict_proba(&features))
            .map_err(|error| crate::support::errors::ml(py, error))?;

        Ok(array::to_numpy_owned(py, probabilities)?.into_any().unbind())
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

    #[getter]
    fn variance_smoothing(&self) -> f64 {
        self.config.variance_smoothing()
    }
}
