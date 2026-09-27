use atlas_random::AtlasRng;
use pyo3::{prelude::*, types::PySequence};

use crate::{array, gil};

#[pyclass(module = "atlas._native")]
pub(crate) struct Generator {
    #[allow(dead_code, reason = "random distribution methods are registered incrementally")]
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
}

impl Generator {
    fn sample(
        &mut self,
        py: Python<'_>,
        shape: &Bound<'_, PyAny>,
        sample: impl FnOnce(
            &[usize],
            &mut AtlasRng,
        ) -> atlas_random::AtlasRandomResult<atlas_ndarray::NDArray<f64>>
        + Send,
    ) -> PyResult<Py<PyAny>> {
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
