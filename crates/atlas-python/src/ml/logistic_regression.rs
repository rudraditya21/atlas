use pyo3::prelude::*;

use crate::support::{arrays as array, gil};

const FIT_OP: &str = "binary_logistic_regression_fit";
const PREDICT_PROBA_OP: &str = "binary_logistic_regression_predict_proba";
const PREDICT_OP: &str = "binary_logistic_regression_predict";

#[pyclass(module = "atlas._native")]
pub(crate) struct BinaryLogisticRegression {
    config: atlas_ml::LogisticRegressionConfig,
    model: super::model::NativeModel<atlas_ml::BinaryLogisticRegression>,
}

#[pymethods]
impl BinaryLogisticRegression {
    #[new]
    #[pyo3(signature = (
        learning_rate = 0.1,
        max_iterations = 1000,
        convergence_tolerance = 1e-6,
        l2_regularization = 0.0
    ))]
    fn new(
        py: Python<'_>,
        learning_rate: f64,
        max_iterations: usize,
        convergence_tolerance: f64,
        l2_regularization: f64,
    ) -> PyResult<Self> {
        let config = atlas_ml::LogisticRegressionConfig::new(
            learning_rate,
            max_iterations,
            convergence_tolerance,
        )
        .and_then(|config| config.with_l2_regularization(l2_regularization))
        .map_err(|error| crate::support::errors::ml(py, error))?;

        Ok(Self { config, model: super::model::NativeModel::new() })
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
            atlas_ml::BinaryLogisticRegression::fit(&features, &labels, config)
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
}
