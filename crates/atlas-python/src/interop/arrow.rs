use pyo3::{exceptions::PyTypeError, prelude::*, types::PyModule};

use crate::{array, python_dtype::with_dtype};

pub(crate) fn to_arrow_primitive(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let dtype = array::source_dtype(py, value)?;

    with_dtype!(
        dtype,
        all | T | {
            let value = array::from_numpy(array::readonly_from_python::<T>(py, value)?)?;
            let arrow = atlas_arrow::to_arrow_primitive::<T, _>(&value)
                .map_err(|error| crate::error::arrow(py, error))?;
            let values = atlas_arrow::from_arrow_primitive::<T>(&arrow)
                .map_err(|error| crate::error::arrow(py, error))?;
            let values = array::to_numpy_owned(py, values)?.into_any();

            Ok(PyModule::import(py, "pyarrow")?.getattr("array")?.call1((values,))?.unbind())
        }
    )
}

pub(crate) fn from_arrow_primitive(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let pyarrow = PyModule::import(py, "pyarrow")?;
    if !value.is_instance(&pyarrow.getattr("Array")?)? {
        return Err(PyTypeError::new_err("expected a pyarrow.Array"));
    }
    if value.getattr("null_count")?.extract::<usize>()? != 0 {
        return Err(crate::error::arrow(
            py,
            atlas_arrow::AtlasArrowError::NullValues { op: "from_arrow_primitive" },
        ));
    }

    let values = value.call_method1("to_numpy", (false,))?;
    let dtype = array::source_dtype(py, &values)?;

    with_dtype!(
        dtype,
        all | T | {
            let values = array::from_numpy(array::readonly_from_python::<T>(py, &values)?)?;
            let arrow = atlas_arrow::to_arrow_primitive::<T, _>(&values)
                .map_err(|error| crate::error::arrow(py, error))?;
            let values = atlas_arrow::from_arrow_primitive::<T>(&arrow)
                .map_err(|error| crate::error::arrow(py, error))?;

            Ok(array::to_numpy_owned(py, values)?.into_any().unbind())
        }
    )
}
