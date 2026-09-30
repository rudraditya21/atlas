use std::collections::HashSet;

use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyList, PyModule},
};

use crate::support::arrays as array;

pub(crate) fn to_arrow_primitive(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> crate::support::results::PyObjectResult {
    array::source_dtype(py, value)?;
    let rank = value.getattr("ndim")?.extract::<usize>()?;
    if rank != 1 {
        return Err(crate::support::errors::arrow(
            py,
            atlas_arrow::AtlasArrowError::InvalidInputRank {
                op: "to_arrow_primitive",
                expected: "rank-1 array",
                rank,
            },
        ));
    }

    PyModule::import(py, "pyarrow")?.getattr("array")?.call1((value,)).map(|values| values.unbind())
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
    array::source_dtype(py, &values)?;
    Ok(values.unbind())
}

pub(crate) fn to_arrow_record_batch(
    py: Python<'_>,
    matrix: &Bound<'_, PyAny>,
    column_names: Vec<String>,
) -> crate::support::results::PyObjectResult {
    let mut unique_names = HashSet::with_capacity(column_names.len());
    if column_names.iter().any(|name| !unique_names.insert(name)) {
        return Err(PyValueError::new_err("Arrow record batch column names must be unique"));
    }

    array::source_dtype(py, matrix)?;
    let shape = matrix.getattr("shape")?.extract::<Vec<usize>>()?;
    if shape.len() != 2 {
        return Err(crate::support::errors::arrow(
            py,
            atlas_arrow::AtlasArrowError::InvalidInputRank {
                op: "to_arrow_record_batch",
                expected: "rank-2 matrix",
                rank: shape.len(),
            },
        ));
    }
    if column_names.len() != shape[1] {
        return Err(crate::support::errors::arrow(
            py,
            atlas_arrow::AtlasArrowError::ColumnNameCountMismatch {
                op: "to_arrow_record_batch",
                expected: shape[1],
                actual: column_names.len(),
            },
        ));
    }

    let pyarrow = PyModule::import(py, "pyarrow")?;
    let columns = PyList::empty(py);
    let fields = PyList::empty(py);
    let transposed = matrix.getattr("T")?;
    for (column_index, name) in column_names.iter().enumerate() {
        let column = pyarrow.getattr("array")?.call1((transposed.get_item(column_index)?,))?;
        fields.append(pyarrow.getattr("field")?.call1((
            name,
            column.getattr("type")?,
            false,
        ))?)?;
        columns.append(column)?;
    }

    let schema = pyarrow.getattr("schema")?.call1((fields,))?;
    pyarrow
        .getattr("RecordBatch")?
        .getattr("from_arrays")?
        .call1((columns, py.None(), schema))
        .map(|batch| batch.unbind())
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
    array::source_dtype(py, &matrix)?;
    Ok(matrix.unbind())
}
