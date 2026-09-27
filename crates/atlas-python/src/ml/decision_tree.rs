use pyo3::prelude::*;

use crate::{array, gil};

#[pyclass(module = "atlas._native")]
pub(crate) struct BinaryGiniSplit {
    impurity: f64,
    left_count: usize,
    right_count: usize,
}

#[pymethods]
impl BinaryGiniSplit {
    #[getter]
    fn impurity(&self) -> f64 {
        self.impurity
    }

    #[getter]
    fn left_count(&self) -> usize {
        self.left_count
    }

    #[getter]
    fn right_count(&self) -> usize {
        self.right_count
    }
}

pub(crate) fn evaluate_binary_gini_split(
    py: Python<'_>,
    feature_values: &Bound<'_, PyAny>,
    labels: &Bound<'_, PyAny>,
    threshold: f64,
) -> PyResult<Py<BinaryGiniSplit>> {
    let feature_values = array::vector_f64(py, feature_values)?;
    let labels = array::label_vector_usize(py, labels)?;
    let split = gil::without_gil(py, move || {
        atlas_ml::evaluate_binary_gini_split(&feature_values, &labels, threshold)
    })
    .map_err(|error| crate::error::ml(py, error))?;

    Py::new(
        py,
        BinaryGiniSplit {
            impurity: split.impurity(),
            left_count: split.left_count(),
            right_count: split.right_count(),
        },
    )
}
