use atlas_ndarray::{ArrayElement, NDArray};
use numpy::Element;
use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyDict, PyList, PyModule},
};

use crate::support::{arrays as array, dtypes::with_dtype};

pub(crate) fn to_arrow_primitive(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> crate::support::results::PyObjectResult {
    let dtype = array::source_dtype(py, value)?;

    with_dtype!(
        dtype,
        all | T | {
            let value = array::from_numpy(array::readonly_from_python::<T>(py, value)?)?;
            let arrow = atlas_arrow::to_arrow_primitive::<T, _>(&value)
                .map_err(|error| crate::support::errors::arrow(py, error))?;
            let values = atlas_arrow::from_arrow_primitive::<T>(&arrow)
                .map_err(|error| crate::support::errors::arrow(py, error))?;
            let values = array::to_numpy_owned(py, values)?.into_any();

            Ok(PyModule::import(py, "pyarrow")?.getattr("array")?.call1((values,))?.unbind())
        }
    )
}

pub(crate) fn from_arrow_primitive(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> crate::support::results::PyObjectResult {
    let pyarrow = PyModule::import(py, "pyarrow")?;
    if !value.is_instance(&pyarrow.getattr("Array")?)? {
        return Err(PyTypeError::new_err("expected a pyarrow.Array"));
    }
    if value.getattr("null_count")?.extract::<usize>()? != 0 {
        return Err(crate::support::errors::arrow(
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
                .map_err(|error| crate::support::errors::arrow(py, error))?;
            let values = atlas_arrow::from_arrow_primitive::<T>(&arrow)
                .map_err(|error| crate::support::errors::arrow(py, error))?;

            Ok(array::to_numpy_owned(py, values)?.into_any().unbind())
        }
    )
}

pub(crate) fn to_arrow_record_batch(
    py: Python<'_>,
    matrix: &Bound<'_, PyAny>,
    column_names: Vec<String>,
) -> crate::support::results::PyObjectResult {
    let dtype = array::source_dtype(py, matrix)?;

    with_dtype!(
        dtype,
        all | T | {
            let matrix = array::from_numpy(array::readonly_from_python::<T>(py, matrix)?)?;
            let names = column_names.iter().map(String::as_str).collect::<Vec<_>>();
            let batch = atlas_arrow::to_arrow_record_batch(&matrix, &names)
                .map_err(|error| crate::support::errors::arrow(py, error))?;
            let matrix = atlas_arrow::from_arrow_record_batch::<T>(&batch)
                .map_err(|error| crate::support::errors::arrow(py, error))?;

            record_batch_from_matrix(py, &matrix, &column_names)
        }
    )
}

pub(crate) fn from_arrow_record_batch(
    py: Python<'_>,
    batch: &Bound<'_, PyAny>,
) -> crate::support::results::PyObjectResult {
    let pyarrow = PyModule::import(py, "pyarrow")?;
    if !batch.is_instance(&pyarrow.getattr("RecordBatch")?)? {
        return Err(PyTypeError::new_err("expected a pyarrow.RecordBatch"));
    }

    let column_count = batch.getattr("num_columns")?.extract::<usize>()?;
    let row_count = batch.getattr("num_rows")?.extract::<usize>()?;
    if column_count == 0 {
        return PyModule::import(py, "numpy")?
            .getattr("empty")?
            .call1(((row_count, 0),))
            .map(|matrix| matrix.unbind());
    }

    let columns = PyList::empty(py);
    let mut dtype = None;
    for column_index in 0..column_count {
        let column = batch.call_method1("column", (column_index,))?;
        if column.getattr("null_count")?.extract::<usize>()? != 0 {
            return Err(crate::support::errors::arrow(
                py,
                atlas_arrow::AtlasArrowError::NullValues { op: "from_arrow_record_batch" },
            ));
        }
        let column_dtype = column.getattr("type")?.str()?.extract::<String>()?;
        if let Some(dtype) = &dtype {
            if dtype != &column_dtype {
                return Err(PyTypeError::new_err(
                    "Arrow record batch columns must share a primitive dtype",
                ));
            }
        } else {
            dtype = Some(column_dtype);
        }
        columns.append(column.call_method1("to_numpy", (false,))?)?;
    }

    let matrix = PyModule::import(py, "numpy")?.getattr("column_stack")?.call1((columns,))?;
    let dtype = array::source_dtype(py, &matrix)?;
    let names = (0..column_count).map(|index| format!("column_{index}")).collect::<Vec<_>>();

    with_dtype!(
        dtype,
        all | T | {
            let matrix = array::from_numpy(array::readonly_from_python::<T>(py, &matrix)?)?;
            let names = names.iter().map(String::as_str).collect::<Vec<_>>();
            let batch = atlas_arrow::to_arrow_record_batch(&matrix, &names)
                .map_err(|error| crate::support::errors::arrow(py, error))?;
            let matrix = atlas_arrow::from_arrow_record_batch::<T>(&batch)
                .map_err(|error| crate::support::errors::arrow(py, error))?;

            Ok(array::to_numpy_owned(py, matrix)?.into_any().unbind())
        }
    )
}

fn record_batch_from_matrix<T>(
    py: Python<'_>,
    matrix: &NDArray<T>,
    column_names: &[String],
) -> crate::support::results::PyObjectResult
where
    T: ArrayElement + Element,
{
    let pyarrow = PyModule::import(py, "pyarrow")?;
    let columns = PyDict::new(py);
    for column_index in 0..matrix.shape()[1] {
        let values = (0..matrix.shape()[0])
            .map(|row_index| matrix.data()[row_index * matrix.shape()[1] + column_index])
            .collect();
        let values = array::to_numpy_owned(
            py,
            NDArray::from_shape_vec([matrix.shape()[0]], values)
                .map_err(|error| crate::support::errors::ndarray(py, error))?,
        )?
        .into_any();
        columns
            .set_item(&column_names[column_index], pyarrow.getattr("array")?.call1((values,))?)?;
    }

    pyarrow
        .getattr("RecordBatch")?
        .getattr("from_pydict")?
        .call1((columns,))
        .map(|batch| batch.unbind())
}
