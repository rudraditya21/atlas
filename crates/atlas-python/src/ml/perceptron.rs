use pyo3::prelude::*;

use crate::{array, gil, model_support};

const FIT_OP: &str = "binary_perceptron_fit";
const PREDICT_OP: &str = "binary_perceptron_predict";

#[pyclass(module = "atlas._native")]
pub(crate) struct BinaryPerceptron {
    config: atlas_ml::PerceptronConfig,
    model: model_support::NativeModel<atlas_ml::BinaryPerceptron>,
}

#[pymethods]
impl BinaryPerceptron {
    #[new]
    #[pyo3(signature = (learning_rate = 0.1, max_iterations = 1000, shuffle_seed = None))]
    fn new(
        py: Python<'_>,
        learning_rate: f64,
        max_iterations: usize,
        shuffle_seed: Option<u64>,
    ) -> PyResult<Self> {
        let config = atlas_ml::PerceptronConfig::new(learning_rate, max_iterations)
            .map_err(|error| crate::error::ml(py, error))?
            .with_shuffle_policy(match shuffle_seed {
                Some(seed) => atlas_ml::PerceptronShufflePolicy::Seeded(seed),
                None => atlas_ml::PerceptronShufflePolicy::Disabled,
            });

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
        let model = gil::without_gil(py, move || {
            atlas_ml::BinaryPerceptron::fit(&features, &labels, config)
        })
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
