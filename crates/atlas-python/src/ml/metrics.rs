use pyo3::prelude::*;

use crate::{array, gil};

pub(crate) fn accuracy(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    predicted: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let actual = array::label_vector_usize(py, actual)?;
    let predicted = array::label_vector_usize(py, predicted)?;
    gil::without_gil(py, move || atlas_ml::classification_accuracy(&actual, &predicted))
        .map_err(|error| crate::error::ml(py, error))
}

pub(crate) fn log_loss(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    probabilities: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let actual = array::label_vector_usize(py, actual)?;
    let probabilities = array::vector_f64(py, probabilities)?;
    gil::without_gil(py, move || atlas_ml::binary_log_loss(&actual, &probabilities))
        .map_err(|error| crate::error::ml(py, error))
}

pub(crate) fn mean_absolute_error(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    predicted: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    regression_metric(py, actual, predicted, atlas_ml::mean_absolute_error)
}

pub(crate) fn mean_squared_error(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    predicted: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    regression_metric(py, actual, predicted, atlas_ml::mean_squared_error)
}

pub(crate) fn r_squared(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    predicted: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    regression_metric(py, actual, predicted, atlas_ml::coefficient_of_determination)
}

fn regression_metric(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    predicted: &Bound<'_, PyAny>,
    metric: impl FnOnce(
        &atlas_ndarray::NDArray<f64>,
        &atlas_ndarray::NDArray<f64>,
    ) -> atlas_ml::AtlasMlResult<f64>
    + Send,
) -> PyResult<f64> {
    let actual = array::vector_f64(py, actual)?;
    let predicted = array::vector_f64(py, predicted)?;
    gil::without_gil(py, move || metric(&actual, &predicted))
        .map_err(|error| crate::error::ml(py, error))
}
