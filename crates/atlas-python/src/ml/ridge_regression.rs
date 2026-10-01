use pyo3::{prelude::*, types::PyDict};

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

    fn copy(&self) -> Self {
        Self { config: self.config, model: super::model::NativeModel::new() }
    }

    fn __copy__(&self) -> Self {
        self.copy()
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

    #[pyo3(signature = (deep = true))]
    fn get_params(&self, py: Python<'_>, deep: bool) -> PyResult<Py<PyDict>> {
        let _ = deep;
        let parameters = PyDict::new(py);
        parameters.set_item("l2_regularization", self.config.l2_regularization())?;
        Ok(parameters.unbind())
    }

    #[pyo3(signature = (**kwargs))]
    fn set_params<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'_>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let mut l2_regularization = slf.config.l2_regularization();
        if let Some(kwargs) = kwargs {
            for (name, value) in kwargs.iter() {
                match name.extract::<&str>()? {
                    "l2_regularization" => l2_regularization = value.extract()?,
                    name => return Err(super::model::unexpected_parameter(name)),
                }
            }
        }
        let config = atlas_ml::RidgeRegressionConfig::new(l2_regularization)
            .map_err(|error| crate::support::errors::ml(py, error))?;
        if config != slf.config {
            slf.config = config;
            slf.model.clear();
        }
        Ok(slf)
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        targets: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let config = slf.config;
        let model = super::model::with_regression_fit_inputs(
            py,
            features,
            targets,
            FIT_OP,
            |features, targets| {
                gil::without_gil(py, || atlas_ml::RidgeRegression::fit(features, targets, config))
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

    fn score(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        targets: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        let model = self.model.fitted(py, PREDICT_OP)?;
        super::model::with_regression_fit_inputs(
            py,
            features,
            targets,
            PREDICT_OP,
            |features, targets| gil::without_gil(py, || model.score(features, targets)),
        )
    }

    #[getter]
    fn l2_regularization(&self) -> f64 {
        self.config.l2_regularization()
    }
}
