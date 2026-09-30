use atlas_random::{AtlasRng, RandomSource};
use numpy::{Element, PyArray1, PyArrayDyn, PyArrayMethods};
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
        self.sample(py, shape, |output, rng| rng.fill_uniform(0.0_f64, 1.0, output))
    }

    fn randn(&mut self, py: Python<'_>, shape: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.sample(py, shape, |output, rng| rng.fill_normal(0.0_f64, 1.0, output))
    }

    #[pyo3(signature = (shape, low = 0.0, high = 1.0))]
    fn uniform(
        &mut self,
        py: Python<'_>,
        shape: &Bound<'_, PyAny>,
        low: f64,
        high: f64,
    ) -> PyResult<Py<PyAny>> {
        self.sample(py, shape, |output, rng| rng.fill_uniform(low, high, output))
    }

    #[pyo3(signature = (shape, mean = 0.0, stddev = 1.0))]
    fn normal(
        &mut self,
        py: Python<'_>,
        shape: &Bound<'_, PyAny>,
        mean: f64,
        stddev: f64,
    ) -> PyResult<Py<PyAny>> {
        self.sample(py, shape, |output, rng| rng.fill_normal(mean, stddev, output))
    }

    fn randint(
        &mut self,
        py: Python<'_>,
        shape: &Bound<'_, PyAny>,
        low: i64,
        high: i64,
    ) -> PyResult<Py<PyAny>> {
        self.sample(py, shape, |output, rng| rng.fill_uniform(low, high, output))
    }

    #[pyo3(signature = (shape, probability = 0.5))]
    fn bernoulli(
        &mut self,
        py: Python<'_>,
        shape: &Bound<'_, PyAny>,
        probability: f64,
    ) -> PyResult<Py<PyAny>> {
        self.sample(py, shape, |output, rng| rng.fill_bernoulli(probability, output))
    }

    fn categorical(
        &mut self,
        py: Python<'_>,
        shape: &Bound<'_, PyAny>,
        weights: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let weights = weights.extract::<Vec<f64>>()?;
        self.sample(py, shape, |output, rng| atlas_random::fill_categorical(output, &weights, rng))
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
                let output = PyArray1::<$ty>::zeros(py, [sample_count], false);
                let result = {
                    let mut destination = output.readwrite();
                    atlas_random::fill_choice(&values, destination.as_slice_mut()?, &mut self.rng)
                };
                result.map_err(|error| crate::support::errors::random(py, error))?;
                Ok(output.into_any().unbind())
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
        let output = PyArray1::<usize>::zeros(py, [size], false);
        {
            let mut destination = output.readwrite();
            let values = destination.as_slice_mut()?;
            for (index, value) in values.iter_mut().enumerate() {
                *value = index;
            }
            self.rng.shuffle(values);
        }

        Ok(output.into_any().unbind())
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
        fill: impl FnOnce(&mut [T], &mut AtlasRng) -> atlas_random::AtlasRandomResult<()>,
    ) -> PyResult<Py<PyAny>>
    where
        T: atlas_ndarray::ArrayElement + Element,
    {
        let shape = shape_values(shape)?;
        atlas_ndarray::checked_element_count(&shape)
            .map_err(|error| crate::support::errors::random(py, error.into()))?;
        let output = PyArrayDyn::<T>::zeros(py, shape.as_slice(), false);
        let result = {
            let mut destination = output.readwrite();
            fill(destination.as_slice_mut()?, &mut self.rng)
        };
        result.map_err(|error| crate::support::errors::random(py, error))?;

        Ok(output.into_any().unbind())
    }
}

fn shape_values(shape: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if shape.is_instance_of::<PySequence>() {
        return shape.extract();
    }

    shape.extract::<usize>().map(|size| vec![size])
}
