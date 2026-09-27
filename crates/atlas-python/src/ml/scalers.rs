use pyo3::prelude::*;

use crate::support::{arrays as array, gil};

const STANDARD_TRANSFORM_OP: &str = "standard_scaler_transform";
const STANDARD_INVERSE_TRANSFORM_OP: &str = "standard_scaler_inverse_transform";
const MIN_MAX_TRANSFORM_OP: &str = "min_max_scaler_transform";
const MIN_MAX_INVERSE_TRANSFORM_OP: &str = "min_max_scaler_inverse_transform";

#[pyclass(module = "atlas._native")]
pub(crate) struct StandardScaler {
    model: super::model::NativeModel<atlas_ml::StandardScaler>,
}

#[pymethods]
impl StandardScaler {
    #[new]
    fn new() -> Self {
        Self { model: super::model::NativeModel::new() }
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let features = array::feature_matrix_f64(py, features)?;
        let model = gil::without_gil(py, move || atlas_ml::StandardScaler::fit(&features))
            .map_err(|error| crate::support::errors::ml(py, error))?;

        slf.model.replace(model);
        Ok(slf)
    }

    fn fit_transform(
        &mut self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
    ) -> crate::support::results::PyObjectResult {
        let features = array::feature_matrix_f64(py, features)?;
        let (model, transformed) =
            gil::without_gil(py, move || atlas_ml::StandardScaler::fit_transform(&features))
                .map_err(|error| crate::support::errors::ml(py, error))?;

        self.model.replace(model);
        Ok(array::to_numpy_owned(py, transformed)?.into_any().unbind())
    }

    fn transform(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
    ) -> crate::support::results::PyObjectResult {
        transform(py, features, &self.model, STANDARD_TRANSFORM_OP, |model, features| {
            model.transform(features)
        })
    }

    fn inverse_transform(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
    ) -> crate::support::results::PyObjectResult {
        transform(py, features, &self.model, STANDARD_INVERSE_TRANSFORM_OP, |model, features| {
            model.inverse_transform(features)
        })
    }
}

#[pyclass(module = "atlas._native")]
pub(crate) struct MinMaxScaler {
    output_minimum: f64,
    output_maximum: f64,
    model: super::model::NativeModel<atlas_ml::MinMaxScaler>,
}

#[pymethods]
impl MinMaxScaler {
    #[new]
    #[pyo3(signature = (output_minimum = 0.0, output_maximum = 1.0))]
    fn new(output_minimum: f64, output_maximum: f64) -> Self {
        Self { output_minimum, output_maximum, model: super::model::NativeModel::new() }
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let features = array::feature_matrix_f64(py, features)?;
        let output_minimum = slf.output_minimum;
        let output_maximum = slf.output_maximum;
        let model = gil::without_gil(py, move || {
            atlas_ml::MinMaxScaler::fit_with_range(&features, output_minimum, output_maximum)
        })
        .map_err(|error| crate::support::errors::ml(py, error))?;

        slf.model.replace(model);
        Ok(slf)
    }

    fn fit_transform(
        &mut self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
    ) -> crate::support::results::PyObjectResult {
        let features = array::feature_matrix_f64(py, features)?;
        let output_minimum = self.output_minimum;
        let output_maximum = self.output_maximum;
        let (model, transformed) = gil::without_gil(py, move || {
            let model =
                atlas_ml::MinMaxScaler::fit_with_range(&features, output_minimum, output_maximum)?;
            let transformed = model.transform(&features)?;
            Ok::<_, atlas_ml::AtlasMlError>((model, transformed))
        })
        .map_err(|error| crate::support::errors::ml(py, error))?;

        self.model.replace(model);
        Ok(array::to_numpy_owned(py, transformed)?.into_any().unbind())
    }

    fn transform(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
    ) -> crate::support::results::PyObjectResult {
        transform(py, features, &self.model, MIN_MAX_TRANSFORM_OP, |model, features| {
            model.transform(features)
        })
    }

    fn inverse_transform(
        &self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
    ) -> crate::support::results::PyObjectResult {
        transform(py, features, &self.model, MIN_MAX_INVERSE_TRANSFORM_OP, |model, features| {
            model.inverse_transform(features)
        })
    }

    #[getter]
    fn output_minimum(&self) -> f64 {
        self.output_minimum
    }

    #[getter]
    fn output_maximum(&self) -> f64 {
        self.output_maximum
    }
}

fn transform<T: Sync>(
    py: Python<'_>,
    features: &Bound<'_, PyAny>,
    model: &super::model::NativeModel<T>,
    operation: &'static str,
    operation_fn: impl FnOnce(
        &T,
        &atlas_ndarray::NDArray<f64>,
    ) -> atlas_ml::AtlasMlResult<atlas_ndarray::NDArray<f64>>
    + Send,
) -> crate::support::results::PyObjectResult {
    let features = super::model::predict_features(py, features, operation)?;
    let model = model.fitted(py, operation)?;
    let transformed = gil::without_gil(py, move || operation_fn(model, &features))
        .map_err(|error| crate::support::errors::ml(py, error))?;

    Ok(array::to_numpy_owned(py, transformed)?.into_any().unbind())
}
