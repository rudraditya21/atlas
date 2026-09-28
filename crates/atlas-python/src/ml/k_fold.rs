use numpy::PyArray1;
use pyo3::prelude::*;

use crate::support::{arrays as array, gil};

#[pyclass(module = "atlas._native")]
pub(crate) struct KFold {
    train_indices: Py<PyAny>,
    validation_indices: Py<PyAny>,
}

#[pymethods]
impl KFold {
    #[getter]
    fn train_indices(&self, py: Python<'_>) -> Py<PyAny> {
        self.train_indices.clone_ref(py)
    }

    #[getter]
    fn validation_indices(&self, py: Python<'_>) -> Py<PyAny> {
        self.validation_indices.clone_ref(py)
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(array::metadata_len(py, &self.train_indices)?
            + array::metadata_len(py, &self.validation_indices)?)
    }
}

pub(crate) fn k_fold_split(
    py: Python<'_>,
    features: &Bound<'_, PyAny>,
    fold_count: usize,
    seed: u64,
) -> PyResult<Vec<Py<KFold>>> {
    let features = array::feature_matrix_f64(py, features)?;
    let folds = gil::without_gil(py, move || atlas_ml::k_fold_split(&features, fold_count, seed))
        .map_err(|error| crate::support::errors::ml(py, error))?;

    output(py, folds)
}

pub(crate) fn stratified_k_fold_split(
    py: Python<'_>,
    features: &Bound<'_, PyAny>,
    labels: &Bound<'_, PyAny>,
    fold_count: usize,
    seed: u64,
) -> PyResult<Vec<Py<KFold>>> {
    let features = array::feature_matrix_f64(py, features)?;
    let labels = array::label_vector_usize(py, labels)?;
    let folds = gil::without_gil(py, move || {
        atlas_ml::stratified_k_fold_split(&features, &labels, fold_count, seed)
    })
    .map_err(|error| crate::support::errors::ml(py, error))?;

    output(py, folds)
}

fn output(py: Python<'_>, folds: Vec<atlas_ml::KFold>) -> PyResult<Vec<Py<KFold>>> {
    folds
        .into_iter()
        .map(|fold| {
            Py::new(
                py,
                KFold {
                    train_indices: PyArray1::from_vec(py, fold.train_indices().to_vec())
                        .into_any()
                        .unbind(),
                    validation_indices: PyArray1::from_vec(py, fold.validation_indices().to_vec())
                        .into_any()
                        .unbind(),
                },
            )
        })
        .collect()
}
