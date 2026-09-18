use atlas_ndarray::{ArrayElement, NDArray};
use numpy::{Element, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyValueError, prelude::*};

pub(crate) fn from_numpy<T>(array: PyReadonlyArrayDyn<'_, T>) -> PyResult<NDArray<T>>
where
    T: ArrayElement + Element,
{
    let array = array.as_array();
    let shape = array.shape().to_vec();
    let data = array.iter().copied().collect();

    NDArray::from_shape_vec(shape, data).map_err(|error| PyValueError::new_err(error.to_string()))
}
