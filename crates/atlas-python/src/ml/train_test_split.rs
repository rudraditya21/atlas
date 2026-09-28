use atlas_ndarray::ArrayElement;
use numpy::Element;
use pyo3::prelude::*;

use crate::support::{arrays as array, dtypes::with_dtype, gil};

#[pyclass(module = "atlas._native")]
pub(crate) struct TrainTestSplit {
    train_features: Py<PyAny>,
    train_targets: Py<PyAny>,
    test_features: Py<PyAny>,
    test_targets: Py<PyAny>,
}

#[pymethods]
impl TrainTestSplit {
    #[getter]
    fn train_features(&self, py: Python<'_>) -> Py<PyAny> {
        self.train_features.clone_ref(py)
    }

    #[getter]
    fn train_targets(&self, py: Python<'_>) -> Py<PyAny> {
        self.train_targets.clone_ref(py)
    }

    #[getter]
    fn test_features(&self, py: Python<'_>) -> Py<PyAny> {
        self.test_features.clone_ref(py)
    }

    #[getter]
    fn test_targets(&self, py: Python<'_>) -> Py<PyAny> {
        self.test_targets.clone_ref(py)
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(array::metadata_len(py, &self.train_features)?
            + array::metadata_len(py, &self.test_features)?)
    }
}

pub(crate) fn train_test_split(
    py: Python<'_>,
    features: &Bound<'_, PyAny>,
    targets: &Bound<'_, PyAny>,
    test_ratio: f64,
    seed: u64,
) -> PyResult<Py<TrainTestSplit>> {
    let features = array::feature_matrix_f64(py, features)?;
    let dtype = array::source_dtype(py, targets)?;

    with_dtype!(
        dtype,
        all | T | {
            let targets = array::from_numpy(array::readonly_from_python::<T>(py, targets)?)?;
            let split = gil::without_gil(py, move || {
                atlas_ml::train_test_split(&features, &targets, test_ratio, seed)
            })
            .map_err(|error| crate::support::errors::ml(py, error))?;
            output(py, split)
        }
    )
}

fn output<T>(py: Python<'_>, split: atlas_ml::TrainTestSplit<T>) -> PyResult<Py<TrainTestSplit>>
where
    T: ArrayElement + Element,
{
    Py::new(
        py,
        TrainTestSplit {
            train_features: array::to_numpy_owned(py, split.train_features().clone())?
                .into_any()
                .unbind(),
            train_targets: array::to_numpy_owned(py, split.train_targets().clone())?
                .into_any()
                .unbind(),
            test_features: array::to_numpy_owned(py, split.test_features().clone())?
                .into_any()
                .unbind(),
            test_targets: array::to_numpy_owned(py, split.test_targets().clone())?
                .into_any()
                .unbind(),
        },
    )
}
