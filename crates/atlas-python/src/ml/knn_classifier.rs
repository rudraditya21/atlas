use pyo3::{exceptions::PyValueError, prelude::*};

use crate::{array, gil, model_support};

const FIT_OP: &str = "knn_classifier_fit";
const PREDICT_PROBA_OP: &str = "knn_classifier_predict_proba";
const PREDICT_OP: &str = "knn_classifier_predict";

#[pyclass(module = "atlas._native")]
pub(crate) struct KnnClassifier {
    config: atlas_ml::KnnConfig,
    model: model_support::NativeModel<atlas_ml::KnnClassifier>,
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
        let config = knn_config(py, k, search_algorithm, weighting, tree_leaf_size)?;

        Ok(Self { config, model: model_support::NativeModel::new() })
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
        labels: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (features, labels) =
            model_support::classifier_fit_inputs(py, features, labels, FIT_OP)?;
        let config = slf.config;
        let model =
            gil::without_gil(py, move || atlas_ml::KnnClassifier::fit(features, labels, config))
                .map_err(|error| crate::error::ml(py, error))?;

        slf.model.replace(model);
        Ok(slf)
    }

    fn predict_proba(&self, py: Python<'_>, features: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let features = model_support::predict_features(py, features, PREDICT_PROBA_OP)?;
        let model = self.model.fitted(py, PREDICT_PROBA_OP)?;
        let probabilities = gil::without_gil(py, move || model.predict_proba(&features))
            .map_err(|error| crate::error::ml(py, error))?;

        Ok(array::to_numpy_owned(py, probabilities)?.into_any().unbind())
    }

    fn predict(&self, py: Python<'_>, features: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let features = model_support::predict_features(py, features, PREDICT_OP)?;
        let model = self.model.fitted(py, PREDICT_OP)?;
        let predictions = gil::without_gil(py, move || model.predict(&features))
            .map_err(|error| crate::error::ml(py, error))?;

        Ok(array::to_numpy_owned(py, predictions)?.into_any().unbind())
    }
}

fn knn_config(
    py: Python<'_>,
    k: usize,
    search_algorithm: &str,
    weighting: &str,
    tree_leaf_size: usize,
) -> PyResult<atlas_ml::KnnConfig> {
    let search_algorithm = match search_algorithm {
        "brute_force" => atlas_ml::KnnSearchAlgorithm::BruteForce,
        "kd_tree" => atlas_ml::KnnSearchAlgorithm::KdTree,
        "ball_tree" => atlas_ml::KnnSearchAlgorithm::BallTree,
        "auto" => atlas_ml::KnnSearchAlgorithm::Auto,
        _ => {
            return Err(PyValueError::new_err(
                "search_algorithm must be 'brute_force', 'kd_tree', 'ball_tree', or 'auto'",
            ));
        }
    };
    let weighting = match weighting {
        "uniform" => atlas_ml::KnnWeighting::Uniform,
        "distance" => atlas_ml::KnnWeighting::Distance,
        _ => return Err(PyValueError::new_err("weighting must be 'uniform' or 'distance'")),
    };

    atlas_ml::KnnConfig::new(k)
        .and_then(|config| config.with_search_algorithm(search_algorithm))
        .and_then(|config| config.with_tree_leaf_size(tree_leaf_size))
        .map(|config| config.with_weighting(weighting))
        .map_err(|error| crate::error::ml(py, error))
}
