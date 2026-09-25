use atlas_ndarray::{ArrayElement, NDArray};
use numpy::{
    Element, PyArray1, PyArrayDescr, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods,
    PyReadonlyArrayDyn, dtype,
};
use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::python_dtype::DType;

pub(crate) fn readonly_from_python<'py, T>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArrayDyn<'py, T>>
where
    T: ArrayElement + Element,
{
    validate_dtype::<T>(py, value)?;
    value.extract().map_err(Into::into)
}

pub(crate) fn from_numpy<T>(array: PyReadonlyArrayDyn<'_, T>) -> PyResult<NDArray<T>>
where
    T: ArrayElement + Element,
{
    let array = array.as_array();
    let shape = array.shape().to_vec();
    let data = array.iter().copied().collect();

    NDArray::from_shape_vec(shape, data)
        .map_err(|error| Python::attach(|py| crate::error::ndarray(py, error)))
}

pub(crate) fn to_numpy_owned<'py, T>(
    py: Python<'py>,
    array: NDArray<T>,
) -> PyResult<Bound<'py, PyArrayDyn<T>>>
where
    T: ArrayElement + Element,
{
    let (data, shape) = array.into_raw_parts();

    PyArray1::from_vec(py, data).reshape(shape)
}

fn validate_dtype<T>(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()>
where
    T: Element,
{
    require_numpy_array(py, value)?;

    let actual = value.getattr("dtype")?.cast_into::<PyArrayDescr>()?;
    if actual.is_equiv_to(&dtype::<T>(py)) {
        return Ok(());
    }

    let dtype_name: String = actual.getattr("name")?.extract()?;
    let reason = match actual.kind() {
        b'O' => "object",
        b'S' | b'U' => "string",
        b'V' => "structured",
        b'c' => "complex",
        _ => return Err(PyTypeError::new_err(format!("unsupported NumPy dtype {dtype_name}"))),
    };

    Err(PyTypeError::new_err(format!("unsupported NumPy {reason} dtype")))
}

pub(crate) fn require_numpy_array(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
    if !is_numpy_array(py, value)? {
        return Err(PyTypeError::new_err("expected a NumPy ndarray"));
    }

    Ok(())
}

pub(crate) fn source_dtype(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<DType> {
    require_numpy_array(py, value)?;
    let name: String = value.getattr("dtype")?.getattr("name")?.extract()?;

    DType::from_numpy_name(&name)
}

pub(crate) fn is_numpy_array(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    value.is_instance(&PyModule::import(py, "numpy")?.getattr("ndarray")?)
}
