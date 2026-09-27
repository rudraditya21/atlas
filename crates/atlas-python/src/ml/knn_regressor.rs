use pyo3::prelude::*;

use crate::{array, gil};

const FIT_OP: &str = "knn_regressor_fit";
const PREDICT_OP: &str = "knn_regressor_predict";

#[pyclass(module = "atlas._native")]
pub(crate) struct KnnRegressor {
    config: atlas_ml::KnnConfig,
    model: super::model::NativeModel<atlas_ml::KnnRegressor>,
}

#[pymethods]
impl KnnRegressor {
    #[new]
    #[pyo3(signature = (
        k,
        search_algorithm = "brute_force",
        weighting = "uniform",
        tree_leaf_size = 1
    ))]
    fn new(
        py: Python<'_>,
        k: usize,
        search_algorithm: &str,
        weighting: &str,
        tree_leaf_size: usize,
    ) -> PyResult<Self> {
        let config = super::knn::config(py, k, search_algorithm, weighting, tree_leaf_size)?;

        Ok(Self { config, model: super::model::NativeModel::new() })
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
        let model =
            gil::without_gil(py, move || atlas_ml::KnnRegressor::fit(features, targets, config))
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
