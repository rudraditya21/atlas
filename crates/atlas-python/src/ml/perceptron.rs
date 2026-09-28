use pyo3::{prelude::*, types::PyDict};

use crate::support::{arrays as array, gil};

const FIT_OP: &str = "binary_perceptron_fit";
const PREDICT_OP: &str = "binary_perceptron_predict";

#[pyclass(module = "atlas._native")]
pub(crate) struct BinaryPerceptron {
    config: atlas_ml::PerceptronConfig,
    model: super::model::NativeModel<atlas_ml::BinaryPerceptron>,
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
            .map_err(|error| crate::support::errors::ml(py, error))?
            .with_shuffle_policy(match shuffle_seed {
                Some(seed) => atlas_ml::PerceptronShufflePolicy::Seeded(seed),
                None => atlas_ml::PerceptronShufflePolicy::Disabled,
            });

        Ok(Self { config, model: super::model::NativeModel::new() })
    }

    fn __repr__(&self) -> String {
        let shuffle_seed = match self.config.shuffle_policy() {
            atlas_ml::PerceptronShufflePolicy::Disabled => "None".to_owned(),
            atlas_ml::PerceptronShufflePolicy::Seeded(seed) => seed.to_string(),
        };
        format!(
            "BinaryPerceptron(learning_rate={}, max_iterations={}, shuffle_seed={shuffle_seed})",
            self.config.learning_rate(),
            self.config.max_iterations(),
        )
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
        parameters.set_item("learning_rate", self.config.learning_rate())?;
        parameters.set_item("max_iterations", self.config.max_iterations())?;
        match self.config.shuffle_policy() {
            atlas_ml::PerceptronShufflePolicy::Disabled => {
                parameters.set_item("shuffle_seed", py.None())?;
            }
            atlas_ml::PerceptronShufflePolicy::Seeded(seed) => {
                parameters.set_item("shuffle_seed", seed)?;
            }
        }
        Ok(parameters.unbind())
    }

    #[pyo3(signature = (**kwargs))]
    fn set_params<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'_>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let mut learning_rate = slf.config.learning_rate();
        let mut max_iterations = slf.config.max_iterations();
        let mut shuffle_seed = match slf.config.shuffle_policy() {
            atlas_ml::PerceptronShufflePolicy::Disabled => None,
            atlas_ml::PerceptronShufflePolicy::Seeded(seed) => Some(seed),
        };
        if let Some(kwargs) = kwargs {
            for (name, value) in kwargs.iter() {
                match name.extract::<&str>()? {
                    "learning_rate" => learning_rate = value.extract()?,
                    "max_iterations" => max_iterations = value.extract()?,
                    "shuffle_seed" => shuffle_seed = value.extract()?,
                    name => return Err(super::model::unexpected_parameter(name)),
                }
            }
        }
        let config = atlas_ml::PerceptronConfig::new(learning_rate, max_iterations)
            .map_err(|error| crate::support::errors::ml(py, error))?
            .with_shuffle_policy(match shuffle_seed {
                Some(seed) => atlas_ml::PerceptronShufflePolicy::Seeded(seed),
                None => atlas_ml::PerceptronShufflePolicy::Disabled,
            });
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
        let model = gil::without_gil(py, move || {
            atlas_ml::BinaryPerceptron::fit(&features, &labels, config)
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
}
