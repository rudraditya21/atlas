use atlas_ndarray::{ArrayElement, NDArray};
use numpy::{Element, PyArrayDescr, PyArrayDescrMethods, PyReadonlyArrayDyn, dtype};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};

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

    NDArray::from_shape_vec(shape, data).map_err(|error| PyValueError::new_err(error.to_string()))
}

fn validate_dtype<T>(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()>
where
    T: Element,
{
    let numpy = PyModule::import(py, "numpy")?;
    if !value.is_instance(&numpy.getattr("ndarray")?)? {
        return Err(PyTypeError::new_err("expected a NumPy ndarray"));
    }

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
