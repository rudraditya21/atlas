use atlas_ndarray::{AtlasNdError, NDArray};
use pyo3::prelude::*;

use crate::{array, gil, python_dtype::with_dtype};

pub(crate) fn put(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    indices: Vec<i64>,
    values: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let dtype = array::source_dtype(py, value)?;

    with_dtype!(
        dtype,
        all | T | {
            let value = array::from_numpy(array::readonly_from_python::<T>(py, value)?)?;
            let values = if array::is_numpy_array(py, values)? {
                array::from_numpy(array::readonly_from_python::<T>(py, values)?)?.flatten()
            } else {
                NDArray::from_shape_vec([1], vec![values.extract::<T>()?])
                    .expect("scalar replacement preserves ndarray invariants")
            };
            let result = gil::without_gil(py, move || put_values(value, &indices, values))
                .map_err(|error| crate::error::ndarray(py, error))?;
            Ok(array::to_numpy_owned(py, result)?.into_any().unbind())
        }
    )
}

fn put_values<T: atlas_ndarray::ArrayElement>(
    value: NDArray<T>,
    indices: &[i64],
    values: NDArray<T>,
) -> Result<NDArray<T>, AtlasNdError> {
    if !indices.is_empty() && values.size() == 0 {
        return Err(AtlasNdError::InvalidArgument {
            op: "put",
            reason: "values must not be empty when indices are provided",
        });
    }

    let shape = value.shape().to_vec();
    let mut flattened = value.flatten();
    let repeated = NDArray::from_shape_vec(
        [indices.len()],
        (0..indices.len()).map(|index| values.data()[index % values.size()]).collect(),
    )
    .expect("put replacement shape is valid");
    flattened.put(indices, &repeated, 0)?;
    let (data, _) = flattened.into_raw_parts();
    NDArray::from_shape_vec(shape, data)
}
