use pyo3::prelude::*;

use crate::support::{arrays as array, gil};

const FIT_OP: &str = "ridge_regression_fit";
const PREDICT_OP: &str = "ridge_regression_predict";

#[pyclass(module = "atlas._native")]
pub(crate) struct RidgeRegression {
    config: atlas_ml::RidgeRegressionConfig,
    model: super::model::NativeModel<atlas_ml::RidgeRegression>,
}

#[pymethods]
impl RidgeRegression {
    #[new]
    #[pyo3(signature = (l2_regularization = 0.0))]
    fn new(py: Python<'_>, l2_regularization: f64) -> PyResult<Self> {
        let config = atlas_ml::RidgeRegressionConfig::new(l2_regularization)
            .map_err(|error| crate::support::errors::ml(py, error))?;

        Ok(Self { config, model: super::model::NativeModel::new() })
    }

    fn __repr__(&self) -> String {
        format!("RidgeRegression(l2_regularization={})", self.config.l2_regularization())
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.model.is_fitted()
    }

    #[getter]
    fn intercept_(&self, py: Python<'_>) -> PyResult<f64> {
        Ok(self.model.fitted(py, FIT_OP)?.intercept())
    }

    #[getter]
    fn coef_(&self, py: Python<'_>) -> crate::support::results::PyObjectResult {
        let coefficients = self.model.fitted(py, FIT_OP)?.coefficients().clone();
        Ok(array::to_numpy_owned(py, coefficients)?.into_any().unbind())
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        targets: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (features, targets) =
            super::model::regression_fit_inputs(py, features, targets, FIT_OP)?;
        let config = slf.config;
        let model = gil::without_gil(py, move || {
            atlas_ml::RidgeRegression::fit(&features, &targets, config)
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

    #[getter]
    fn l2_regularization(&self) -> f64 {
        self.config.l2_regularization()
    }
}
