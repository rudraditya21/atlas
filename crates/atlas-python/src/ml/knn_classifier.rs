use pyo3::{prelude::*, types::PyDict};

use crate::support::{arrays as array, gil};

const FIT_OP: &str = "knn_classifier_fit";
const PREDICT_PROBA_OP: &str = "knn_classifier_predict_proba";
const PREDICT_OP: &str = "knn_classifier_predict";

#[pyclass(module = "atlas._native")]
pub(crate) struct KnnClassifier {
    config: atlas_ml::KnnConfig,
    model: super::model::NativeModel<atlas_ml::KnnClassifier>,
}

#[pymethods]
impl KnnClassifier {
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

    fn __repr__(&self) -> String {
        super::knn::repr("KnnClassifier", self.config)
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
    fn classes_(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(array::to_numpy_usize_vector(py, self.model.fitted(py, PREDICT_OP)?.classes()))
    }

    #[getter]
    fn labels_(&self, py: Python<'_>) -> crate::support::results::PyObjectResult {
        let labels = self.model.fitted(py, PREDICT_OP)?.labels().clone();
        Ok(array::to_numpy_owned(py, labels)?.into_any().unbind())
    }

    #[pyo3(signature = (deep = true))]
    fn get_params(&self, py: Python<'_>, deep: bool) -> PyResult<Py<PyDict>> {
        let _ = deep;
        super::knn::parameters(py, self.config)
    }

    #[pyo3(signature = (**kwargs))]
    fn set_params<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'_>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let mut k = slf.config.k();
        let mut search_algorithm = super::knn::search_algorithm(slf.config).to_owned();
        let mut weighting = super::knn::weighting(slf.config).to_owned();
        let mut tree_leaf_size = slf.config.tree_leaf_size();
        if let Some(kwargs) = kwargs {
            for (name, value) in kwargs.iter() {
                match name.extract::<&str>()? {
                    "k" => k = value.extract()?,
                    "search_algorithm" => search_algorithm = value.extract()?,
                    "weighting" => weighting = value.extract()?,
                    "tree_leaf_size" => tree_leaf_size = value.extract()?,
                    name => return Err(super::model::unexpected_parameter(name)),
                }
            }
        }
        let config = super::knn::config(py, k, &search_algorithm, &weighting, tree_leaf_size)?;
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
        labels: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (features, labels) = super::model::classifier_fit_inputs(py, features, labels, FIT_OP)?;
        let config = slf.config;
        let model =
            gil::without_gil(py, move || atlas_ml::KnnClassifier::fit(features, labels, config))
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

    fn fit_predict(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        labels: &Bound<'_, PyAny>,
    ) -> crate::support::results::PyObjectResult {
        let (features, labels) = super::model::classifier_fit_inputs(py, features, labels, FIT_OP)?;
        let queries = features.clone();
        let config = slf.config;
        let model =
            gil::without_gil(py, move || atlas_ml::KnnClassifier::fit(features, labels, config))
                .map_err(|error| crate::support::errors::ml(py, error))?;
        let predictions = gil::without_gil(py, || model.predict(&queries))
            .map_err(|error| crate::support::errors::ml(py, error))?;
        slf.model.replace(model);
        Ok(array::to_numpy_owned(py, predictions)?.into_any().unbind())
    }
}
