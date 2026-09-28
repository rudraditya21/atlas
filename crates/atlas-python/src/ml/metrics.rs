use numpy::PyArray1;
use pyo3::{exceptions::PyKeyError, prelude::*, types::PyTuple};

use crate::support::{arrays as array, gil};

#[pyclass(module = "atlas._native")]
pub(crate) struct ConfusionMatrix {
    classes: Py<PyAny>,
    counts: Py<PyAny>,
}

#[pymethods]
impl ConfusionMatrix {
    #[getter]
    fn classes(&self, py: Python<'_>) -> Py<PyAny> {
        self.classes.clone_ref(py)
    }

    #[getter]
    fn counts(&self, py: Python<'_>) -> Py<PyAny> {
        self.counts.clone_ref(py)
    }

    #[getter]
    fn dtype(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        array::metadata_dtype(py, &self.counts)
    }

    #[getter]
    fn shape(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        array::metadata_shape(py, &self.counts)
    }

    #[getter]
    fn ndim(&self, py: Python<'_>) -> PyResult<usize> {
        array::metadata_ndim(py, &self.counts)
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        array::metadata_len(py, &self.classes)
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Ok(other) = other.extract::<PyRef<'_, Self>>() else {
            return Ok(false);
        };
        Ok(array::values_equal(py, &self.classes, &other.classes)?
            && array::values_equal(py, &self.counts, &other.counts)?)
    }
}

#[pyclass(module = "atlas._native")]
pub(crate) struct ClassificationReport {
    accuracy: f64,
    precision: f64,
    recall: f64,
    f1_score: f64,
}

#[pymethods]
impl ClassificationReport {
    #[getter]
    fn accuracy(&self) -> f64 {
        self.accuracy
    }

    #[getter]
    fn precision(&self) -> f64 {
        self.precision
    }

    #[getter]
    fn recall(&self) -> f64 {
        self.recall
    }

    #[getter]
    fn f1_score(&self) -> f64 {
        self.f1_score
    }

    fn __len__(&self) -> usize {
        4
    }

    fn __getitem__(&self, key: &str) -> PyResult<f64> {
        match key {
            "accuracy" => Ok(self.accuracy),
            "precision" => Ok(self.precision),
            "recall" => Ok(self.recall),
            "f1_score" => Ok(self.f1_score),
            _ => Err(PyKeyError::new_err(key.to_owned())),
        }
    }

    fn keys(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyTuple::new(py, ["accuracy", "precision", "recall", "f1_score"])?.into_any().unbind())
    }

    fn values(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyTuple::new(py, [self.accuracy, self.precision, self.recall, self.f1_score])?
            .into_any()
            .unbind())
    }

    fn items(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyTuple::new(
            py,
            [
                ("accuracy", self.accuracy),
                ("precision", self.precision),
                ("recall", self.recall),
                ("f1_score", self.f1_score),
            ],
        )?
        .into_any()
        .unbind())
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.keys(py)?.bind(py).call_method0("__iter__").map(|iterator| iterator.unbind())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Ok(other) = other.extract::<PyRef<'_, Self>>() else {
            return Ok(false);
        };
        Ok(self.accuracy == other.accuracy
            && self.precision == other.precision
            && self.recall == other.recall
            && self.f1_score == other.f1_score)
    }
}

pub(crate) fn confusion_matrix(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    predicted: &Bound<'_, PyAny>,
) -> PyResult<Py<ConfusionMatrix>> {
    let actual = array::label_vector_usize(py, actual)?;
    let predicted = array::label_vector_usize(py, predicted)?;
    let result = gil::without_gil(py, move || atlas_ml::confusion_matrix(&actual, &predicted))
        .map_err(|error| crate::support::errors::ml(py, error))?;

    Py::new(
        py,
        ConfusionMatrix {
            classes: PyArray1::from_vec(py, result.classes().to_vec()).into_any().unbind(),
            counts: array::to_numpy_owned(py, result.counts().clone())?.into_any().unbind(),
        },
    )
}

pub(crate) fn classification_report(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    predicted: &Bound<'_, PyAny>,
) -> PyResult<Py<ClassificationReport>> {
    let actual = array::label_vector_usize(py, actual)?;
    let predicted = array::label_vector_usize(py, predicted)?;
    let result = gil::without_gil(py, move || atlas_ml::classification_report(&actual, &predicted))
        .map_err(|error| crate::support::errors::ml(py, error))?;

    Py::new(
        py,
        ClassificationReport {
            accuracy: result.accuracy(),
            precision: result.precision(),
            recall: result.recall(),
            f1_score: result.f1_score(),
        },
    )
}

pub(crate) fn accuracy(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    predicted: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let actual = array::label_vector_usize(py, actual)?;
    let predicted = array::label_vector_usize(py, predicted)?;
    gil::without_gil(py, move || atlas_ml::classification_accuracy(&actual, &predicted))
        .map_err(|error| crate::support::errors::ml(py, error))
}

pub(crate) fn log_loss(
    py: Python<'_>,
    actual: &Bound<'_, PyAny>,
    probabilities: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let actual = array::label_vector_usize(py, actual)?;
    let probabilities = array::vector_f64(py, probabilities)?;
    gil::without_gil(py, move || atlas_ml::binary_log_loss(&actual, &probabilities))
        .map_err(|error| crate::support::errors::ml(py, error))
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
        .map_err(|error| crate::support::errors::ml(py, error))
}
