use atlas_ndarray::AtlasNdError;
use pyo3::prelude::*;

use crate::{array, gil, python_dtype::with_dtype};
pub(crate) fn partition(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    kth: &Bound<'_, PyAny>,
    axis: Option<i64>,
) -> PyResult<Py<PyAny>> {
    let dtype = array::source_dtype(py, value)?;
    let kths =
        if let Ok(kth) = kth.extract::<i64>() { vec![kth] } else { kth.extract::<Vec<i64>>()? };

    with_dtype!(
        dtype,
        all | T | {
            let array = array::from_numpy(array::readonly_from_python::<T>(py, value)?)?;
            let result = gil::without_gil(py, move || {
                if let Some(axis) = axis {
                    let kths = normalize_kths(array.shape(), &kths, axis)?;
                    array.partition_many(&kths, axis)
                } else {
                    let array = array.flatten();
                    let kths = normalize_kths(array.shape(), &kths, 0)?;
                    array.partition_many(&kths, 0)
                }
            })
            .map_err(|error| crate::error::ndarray(py, error))?;
            Ok(array::to_numpy_owned(py, result)?.into_any().unbind())
        }
    )
}
pub(crate) fn normalize_kths(
    shape: &[usize],
    kths: &[i64],
    axis: i64,
) -> Result<Vec<usize>, AtlasNdError> {
    let n = shape.len();
    let a = if axis < 0 { axis + n as i64 } else { axis };
    if a < 0 || a as usize >= n {
        return Err(AtlasNdError::InvalidAxis { axis: a, ndim: n });
    }
    let len = shape[a as usize];
    if kths.is_empty() {
        return Err(AtlasNdError::InvalidArgument {
            op: "partition",
            reason: "kth values must not be empty",
        });
    }

    let mut normalized = Vec::with_capacity(kths.len());
    for &kth in kths {
        let kth = if kth < 0 { kth + len as i64 } else { kth };
        if kth < 0 || kth as usize >= len {
            return Err(AtlasNdError::IndexOutOfBounds { axis: a as usize, index: kth, dim: len });
        }
        normalized.push(kth as usize);
    }

    if normalized.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(AtlasNdError::InvalidArgument {
            op: "partition",
            reason: "kth values must be ordered and unique",
        });
    }

    Ok(normalized)
}
