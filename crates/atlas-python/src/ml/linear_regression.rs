use pyo3::prelude::*;

use crate::support::{arrays as array, gil};

const FIT_OP: &str = "linear_regression_fit";
const PREDICT_OP: &str = "linear_regression_predict";

#[pyclass(module = "atlas._native")]
pub(crate) struct LinearRegression {
    model: super::model::NativeModel<atlas_ml::LinearRegression>,
}

#[pymethods]
impl LinearRegression {
    #[new]
    fn new() -> Self {
        Self { model: super::model::NativeModel::new() }
    }

    fn __repr__(&self) -> &'static str {
        "LinearRegression()"
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
        let model = super::model::with_regression_fit_inputs(
            py,
            features,
            targets,
            FIT_OP,
            |features, targets| {
                gil::without_gil(py, || atlas_ml::LinearRegression::fit(features, targets))
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
}
