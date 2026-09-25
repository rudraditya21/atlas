use atlas_ndarray::{NDArray, SortElement};
use pyo3::{prelude::*, types::PyTuple};

use crate::{array, gil, python_dtype::with_dtype};

pub(crate) fn unique(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    return_index: bool,
    return_inverse: bool,
    return_counts: bool,
) -> PyResult<Py<PyAny>> {
    let dtype = array::source_dtype(py, value)?;

    with_dtype!(
        dtype,
        all | T | {
            let array = array::from_numpy(array::readonly_from_python::<T>(py, value)?)?;
            let result = gil::without_gil(py, move || {
                unique_outputs(array, return_index, return_inverse, return_counts)
            });
            let values = array::to_numpy_owned(py, result.values)?.into_any().unbind();
            if !(return_index || return_inverse || return_counts) {
                return Ok(values);
            }

            let mut outputs = vec![values];
            if let Some(indices) = result.indices {
                outputs.push(indices_to_numpy(py, indices)?);
            }
            if let Some(inverse) = result.inverse {
                outputs.push(indices_to_numpy(py, inverse)?);
            }
            if let Some(counts) = result.counts {
                outputs.push(indices_to_numpy(py, counts)?);
            }
            Ok(PyTuple::new(py, outputs)?.into_any().unbind())
        }
    )
}

struct UniqueOutputs<T: SortElement> {
    values: NDArray<T>,
    indices: Option<Vec<i64>>,
    inverse: Option<Vec<i64>>,
    counts: Option<Vec<i64>>,
}

fn unique_outputs<T: SortElement>(
    array: NDArray<T>,
    return_index: bool,
    return_inverse: bool,
    return_counts: bool,
) -> UniqueOutputs<T> {
    let mut order: Vec<_> = (0..array.size()).collect();
    order.sort_by(|&left, &right| array.data()[left].sort_compare(&array.data()[right]));

    let mut values = Vec::new();
    let mut indices = return_index.then(Vec::new);
    let mut inverse = return_inverse.then(|| vec![0; array.size()]);
    let mut counts = return_counts.then(Vec::new);
    for index in order {
        let value = array.data()[index];
        let is_new = values.last().is_none_or(|previous| !value.sort_equal(previous));
        if is_new {
            values.push(value);
            if let Some(indices) = &mut indices {
                indices.push(i64::try_from(index).expect("ndarray indices fit i64"));
            }
            if let Some(counts) = &mut counts {
                counts.push(1);
            }
        } else if let Some(counts) = &mut counts {
            *counts.last_mut().expect("unique group has a count") += 1;
        }
        if let Some(inverse) = &mut inverse {
            inverse[index] = i64::try_from(values.len() - 1).expect("ndarray indices fit i64");
        }
    }

    UniqueOutputs {
        values: NDArray::from_shape_vec([values.len()], values)
            .expect("unique preserves ndarray invariants"),
        indices,
        inverse,
        counts,
    }
}

fn indices_to_numpy(py: Python<'_>, indices: Vec<i64>) -> PyResult<Py<PyAny>> {
    let length = indices.len();
    Ok(array::to_numpy_owned(
        py,
        NDArray::from_shape_vec([length], indices).expect("unique preserves ndarray invariants"),
    )?
    .into_any()
    .unbind())
}
