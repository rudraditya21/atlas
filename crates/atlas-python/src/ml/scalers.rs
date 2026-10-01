use pyo3::{prelude::*, types::PyDict};

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

    fn __repr__(&self) -> &'static str {
        "StandardScaler()"
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.model.is_fitted()
    }

    #[getter]
    fn mean_(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(array::to_numpy_f64_vector(py, self.model.fitted(py, STANDARD_TRANSFORM_OP)?.means()))
    }

    #[getter]
    fn scale_(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(array::to_numpy_f64_vector(py, self.model.fitted(py, STANDARD_TRANSFORM_OP)?.scales()))
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let features = array::readonly_from_python::<f64>(py, features)?;
        let model = array::with_numpy_operand(features, |features| {
            gil::without_gil(py, || atlas_ml::StandardScaler::fit(features))
        })?
        .map_err(|error| crate::support::errors::ml(py, error))?;

        slf.model.replace(model);
        Ok(slf)
    }

    fn fit_transform(
        &mut self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
    ) -> crate::support::results::PyObjectResult {
        let features = array::readonly_from_python::<f64>(py, features)?;
        let (model, transformed) = array::with_numpy_operand(features, |features| {
            gil::without_gil(py, || atlas_ml::StandardScaler::fit_transform(features))
        })?
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

    fn __repr__(&self) -> String {
        format!(
            "MinMaxScaler(output_minimum={}, output_maximum={})",
            self.output_minimum, self.output_maximum
        )
    }

    fn copy(&self) -> Self {
        Self {
            output_minimum: self.output_minimum,
            output_maximum: self.output_maximum,
            model: super::model::NativeModel::new(),
        }
    }

    fn __copy__(&self) -> Self {
        self.copy()
    }

    #[getter]
    fn is_fitted(&self) -> bool {
        self.model.is_fitted()
    }

    #[getter]
    fn data_min_(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(array::to_numpy_f64_vector(py, self.model.fitted(py, MIN_MAX_TRANSFORM_OP)?.minimums()))
    }

    #[getter]
    fn data_max_(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(array::to_numpy_f64_vector(py, self.model.fitted(py, MIN_MAX_TRANSFORM_OP)?.maximums()))
    }

    #[pyo3(signature = (deep = true))]
    fn get_params(&self, py: Python<'_>, deep: bool) -> PyResult<Py<PyDict>> {
        let _ = deep;
        let parameters = PyDict::new(py);
        parameters.set_item("output_minimum", self.output_minimum)?;
        parameters.set_item("output_maximum", self.output_maximum)?;
        Ok(parameters.unbind())
    }

    #[pyo3(signature = (**kwargs))]
    fn set_params<'py>(
        mut slf: PyRefMut<'py, Self>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let mut output_minimum = slf.output_minimum;
        let mut output_maximum = slf.output_maximum;
        if let Some(kwargs) = kwargs {
            for (name, value) in kwargs.iter() {
                match name.extract::<&str>()? {
                    "output_minimum" => output_minimum = value.extract()?,
                    "output_maximum" => output_maximum = value.extract()?,
                    name => return Err(super::model::unexpected_parameter(name)),
                }
            }
        }
        if output_minimum >= output_maximum {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "output_minimum must be less than output_maximum",
            ));
        }
        if output_minimum != slf.output_minimum || output_maximum != slf.output_maximum {
            slf.output_minimum = output_minimum;
            slf.output_maximum = output_maximum;
            slf.model.clear();
        }
        Ok(slf)
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let features = array::readonly_from_python::<f64>(py, features)?;
        let output_minimum = slf.output_minimum;
        let output_maximum = slf.output_maximum;
        let model = array::with_numpy_operand(features, |features| {
            gil::without_gil(py, || {
                atlas_ml::MinMaxScaler::fit_with_range(features, output_minimum, output_maximum)
            })
        })?
        .map_err(|error| crate::support::errors::ml(py, error))?;

        slf.model.replace(model);
        Ok(slf)
    }

    fn fit_transform(
        &mut self,
        py: Python<'_>,
        features: &Bound<'_, PyAny>,
    ) -> crate::support::results::PyObjectResult {
        let features = array::readonly_from_python::<f64>(py, features)?;
        let output_minimum = self.output_minimum;
        let output_maximum = self.output_maximum;
        let (model, transformed) = array::with_numpy_operand(features, |features| {
            gil::without_gil(py, || {
                let model = atlas_ml::MinMaxScaler::fit_with_range(
                    features,
                    output_minimum,
                    output_maximum,
                )?;
                let transformed = model.transform(features)?;
                Ok::<_, atlas_ml::AtlasMlError>((model, transformed))
            })
        })?
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
        &(dyn atlas_ndarray::OperandMetadata<f64> + Sync),
    ) -> atlas_ml::AtlasMlResult<atlas_ndarray::NDArray<f64>>
    + Send,
) -> crate::support::results::PyObjectResult {
    let model = model.fitted(py, operation)?;
    let features = array::readonly_from_python::<f64>(py, features)?;
    let transformed = array::with_numpy_operand(features, |features| {
        gil::without_gil(py, || operation_fn(model, features))
    })?
    .map_err(|error| crate::support::errors::ml(py, error))?;

    Ok(array::to_numpy_owned(py, transformed)?.into_any().unbind())
}
