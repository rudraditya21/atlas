use atlas_random::AtlasRng;
use pyo3::{prelude::*, types::PySequence};

use crate::{array, gil};

#[pyclass(module = "atlas._native")]
pub(crate) struct Generator {
    pub(crate) rng: AtlasRng,
}

#[pymethods]
impl Generator {
    #[new]
    fn new(seed: u64) -> Self {
        Self { rng: AtlasRng::seed_from_u64(seed) }
    }

    fn random(&mut self, py: Python<'_>, shape: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.sample(py, shape, |shape, rng| atlas_random::rand(shape, rng))
    }

    fn randn(&mut self, py: Python<'_>, shape: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.sample(py, shape, |shape, rng| atlas_random::randn(shape, rng))
    }

    #[pyo3(signature = (shape, low = 0.0, high = 1.0))]
    fn uniform(
        &mut self,
        py: Python<'_>,
        shape: &Bound<'_, PyAny>,
        low: f64,
        high: f64,
    ) -> PyResult<Py<PyAny>> {
        self.sample(py, shape, |shape, rng| atlas_random::uniform(shape, low, high, rng))
    }

    #[pyo3(signature = (shape, mean = 0.0, stddev = 1.0))]
    fn normal(
        &mut self,
        py: Python<'_>,
        shape: &Bound<'_, PyAny>,
        mean: f64,
        stddev: f64,
    ) -> PyResult<Py<PyAny>> {
        self.sample(py, shape, |shape, rng| atlas_random::normal(shape, mean, stddev, rng))
    }

    fn randint(
        &mut self,
        py: Python<'_>,
        shape: &Bound<'_, PyAny>,
        low: i64,
        high: i64,
    ) -> PyResult<Py<PyAny>> {
        self.sample(py, shape, |shape, rng| atlas_random::randint(shape, low, high, rng))
    }

    #[pyo3(signature = (shape, probability = 0.5))]
    fn bernoulli(
        &mut self,
        py: Python<'_>,
        shape: &Bound<'_, PyAny>,
        probability: f64,
    ) -> PyResult<Py<PyAny>> {
        self.sample(py, shape, |shape, rng| atlas_random::bernoulli(shape, probability, rng))
    }
}

impl Generator {
    fn sample<T>(
        &mut self,
        py: Python<'_>,
        shape: &Bound<'_, PyAny>,
        sample: impl FnOnce(
            &[usize],
            &mut AtlasRng,
        ) -> atlas_random::AtlasRandomResult<atlas_ndarray::NDArray<T>>
        + Send,
    ) -> PyResult<Py<PyAny>>
    where
        T: atlas_ndarray::ArrayElement + numpy::Element,
    {
        let shape = shape_values(shape)?;
        let result = gil::without_gil(py, move || sample(&shape, &mut self.rng))
            .map_err(|error| crate::error::random(py, error))?;

        Ok(array::to_numpy_owned(py, result)?.into_any().unbind())
    }
}

fn shape_values(shape: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if shape.is_instance_of::<PySequence>() {
        return shape.extract();
    }

    shape.extract::<usize>().map(|size| vec![size])
}
