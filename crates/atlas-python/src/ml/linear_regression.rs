use pyo3::prelude::*;

use crate::{array, gil, model_support};

const FIT_OP: &str = "linear_regression_fit";
const PREDICT_OP: &str = "linear_regression_predict";

#[pyclass(module = "atlas._native")]
pub(crate) struct LinearRegression {
    model: model_support::NativeModel<atlas_ml::LinearRegression>,
}

#[pymethods]
impl LinearRegression {
    #[new]
    fn new() -> Self {
        Self { model: model_support::NativeModel::new() }
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        targets: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (features, targets) =
            model_support::regression_fit_inputs(py, features, targets, FIT_OP)?;
        let model =
            gil::without_gil(py, move || atlas_ml::LinearRegression::fit(&features, &targets))
                .map_err(|error| crate::error::ml(py, error))?;

        slf.model.replace(model);
        Ok(slf)
    }

    fn predict(&self, py: Python<'_>, features: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let features = model_support::predict_features(py, features, PREDICT_OP)?;
        let model = self.model.fitted(py, PREDICT_OP)?;
        let predictions = gil::without_gil(py, move || model.predict(&features))
            .map_err(|error| crate::error::ml(py, error))?;

        Ok(array::to_numpy_owned(py, predictions)?.into_any().unbind())
    }
}
