use atlas_random::AtlasRng;
use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyBytes, PySequence},
};

use crate::support::{arrays as array, gil};

#[pyclass(module = "atlas._native")]
pub(crate) struct Generator {
    pub(crate) rng: AtlasRng,
}

#[pyclass(module = "atlas._native")]
pub(crate) struct GeneratorState {
    rng: AtlasRng,
}

#[pymethods]
impl GeneratorState {
    fn __repr__(&self) -> &'static str {
        "GeneratorState()"
    }
}

#[pymethods]
impl Generator {
    #[new]
    fn new(seed: u64) -> Self {
        Self { rng: AtlasRng::seed_from_u64(seed) }
    }

    #[staticmethod]
    fn from_state(py: Python<'_>, state: &[u8]) -> PyResult<Self> {
        AtlasRng::from_state_bytes(state)
            .map(|rng| Self { rng })
            .map_err(|error| crate::support::errors::random(py, error))
    }

    fn __getnewargs__(&self) -> (u64,) {
        (0,)
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let state =
            self.rng.state_bytes().map_err(|error| crate::support::errors::random(py, error))?;
        Ok(PyBytes::new(py, &state).unbind())
    }

    fn __setstate__(&mut self, py: Python<'_>, state: &[u8]) -> PyResult<()> {
        self.rng = AtlasRng::from_state_bytes(state)
            .map_err(|error| crate::support::errors::random(py, error))?;
        Ok(())
    }

    fn get_state(&self) -> GeneratorState {
        GeneratorState { rng: self.rng.clone() }
    }

    fn set_state(&mut self, state: PyRef<'_, GeneratorState>) {
        self.rng = state.rng.clone();
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

    fn categorical(
        &mut self,
        py: Python<'_>,
        shape: &Bound<'_, PyAny>,
        weights: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let weights = weights.extract::<Vec<f64>>()?;
        self.sample(py, shape, |shape, rng| atlas_random::categorical(shape, &weights, rng))
    }

    fn choice(
        &mut self,
        py: Python<'_>,
        values: &Bound<'_, PyAny>,
        sample_count: usize,
    ) -> PyResult<Py<PyAny>> {
        array::require_numpy_array(py, values)?;
        let dtype: String = values.getattr("dtype")?.getattr("name")?.extract()?;

        macro_rules! apply {
            ($ty:ty) => {{
                let values = array::from_numpy(array::readonly_from_python::<$ty>(py, values)?)?;
                let result = gil::without_gil(py, move || {
                    atlas_random::choice(&values, sample_count, &mut self.rng)
                })
                .map_err(|error| crate::support::errors::random(py, error))?;
                Ok(array::to_numpy_owned(py, result)?.into_any().unbind())
            }};
        }

        match dtype.as_str() {
            "bool" => apply!(bool),
            "int8" => apply!(i8),
            "int16" => apply!(i16),
            "int32" => apply!(i32),
            "int64" => apply!(i64),
            "uint8" => apply!(u8),
            "uint16" => apply!(u16),
            "uint32" => apply!(u32),
            "uint64" => apply!(u64),
            "float32" => apply!(f32),
            "float64" => apply!(f64),
            _ => Err(PyTypeError::new_err(format!("unsupported NumPy dtype {dtype}"))),
        }
    }

    fn permutation(&mut self, py: Python<'_>, size: usize) -> PyResult<Py<PyAny>> {
        let result = gil::without_gil(py, move || atlas_random::permutation(size, &mut self.rng))
            .map_err(|error| crate::support::errors::random(py, error))?;

        Ok(array::to_numpy_owned(py, result)?.into_any().unbind())
    }

    #[pyo3(signature = (values, axis = 0))]
    fn shuffle(
        &mut self,
        py: Python<'_>,
        values: &Bound<'_, PyAny>,
        axis: i64,
    ) -> PyResult<Py<PyAny>> {
        array::require_numpy_array(py, values)?;
        let dtype: String = values.getattr("dtype")?.getattr("name")?.extract()?;

        macro_rules! apply {
            ($ty:ty) => {{
                let values = array::from_numpy(array::readonly_from_python::<$ty>(py, values)?)?;
                let result = gil::without_gil(py, move || {
                    atlas_random::shuffle_axis(&values, axis, &mut self.rng)
                })
                .map_err(|error| crate::support::errors::random(py, error))?;
                Ok(array::to_numpy_owned(py, result)?.into_any().unbind())
            }};
        }

        match dtype.as_str() {
            "bool" => apply!(bool),
            "int8" => apply!(i8),
            "int16" => apply!(i16),
            "int32" => apply!(i32),
            "int64" => apply!(i64),
            "uint8" => apply!(u8),
            "uint16" => apply!(u16),
            "uint32" => apply!(u32),
            "uint64" => apply!(u64),
            "float32" => apply!(f32),
            "float64" => apply!(f64),
            _ => Err(PyTypeError::new_err(format!("unsupported NumPy dtype {dtype}"))),
        }
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
            .map_err(|error| crate::support::errors::random(py, error))?;

        Ok(array::to_numpy_owned(py, result)?.into_any().unbind())
    }
}

fn shape_values(shape: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if shape.is_instance_of::<PySequence>() {
        return shape.extract();
    }

    shape.extract::<usize>().map(|size| vec![size])
}
