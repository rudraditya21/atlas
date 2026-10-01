use atlas_ndarray::{ArrayElement, BorrowedArray, NDArray, OperandMetadata, RuntimeScalar};
use numpy::{
    Element, PyArray1, PyArrayDescr, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods,
    PyReadonlyArrayDyn, PyReadwriteArrayDyn, PyUntypedArrayMethods, dtype,
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};

use crate::support::dtypes::DType;

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
        .map_err(|error| Python::attach(|py| crate::support::errors::ndarray(py, error)))
}

pub(crate) fn with_numpy_operand<T, R>(
    array: PyReadonlyArrayDyn<'_, T>,
    operation: impl FnOnce(&(dyn OperandMetadata<T> + Sync)) -> R,
) -> PyResult<R>
where
    T: ArrayElement + Element,
{
    let view = array.as_array();
    let shape = view.shape().to_vec();
    let strides = positive_element_strides(view.strides(), &shape);

    if let Some(strides) = strides {
        let span_len = storage_span_len(&shape, &strides).ok_or_else(|| {
            Python::attach(|py| {
                crate::support::errors::ndarray(
                    py,
                    atlas_ndarray::AtlasNdError::ShapeOverflow {
                        op: "borrowed NumPy input",
                        shape: shape.clone(),
                    },
                )
            })
        })?;
        let data = if span_len == 0 {
            &[]
        } else {
            // SAFETY: the NumPy read guard keeps the allocation immutably borrowed, and positive
            // strides make every logical address fall between the first element and this span's
            // final element.
            unsafe { std::slice::from_raw_parts(view.as_ptr(), span_len) }
        };
        let operand = BorrowedArray::from_parts(data, &shape, &strides, 0)
            .map_err(|error| Python::attach(|py| crate::support::errors::ndarray(py, error)))?;

        return Ok(operation(&operand));
    }

    let owned = from_numpy(array)?;
    Ok(operation(&owned))
}

pub(crate) fn copy_operand<T, O>(operand: &O) -> atlas_ndarray::AtlasNdResult<NDArray<T>>
where
    T: ArrayElement,
    O: OperandMetadata<T> + ?Sized,
{
    let mut data = Vec::with_capacity(atlas_ndarray::checked_element_count(operand.shape())?);
    atlas_ndarray::try_for_each_logical_span(operand, |span| {
        data.extend_from_slice(span);
        Ok::<(), atlas_ndarray::AtlasNdError>(())
    })?;
    NDArray::from_shape_vec(operand.shape(), data)
}

fn positive_element_strides(strides: &[isize], shape: &[usize]) -> Option<Vec<usize>> {
    strides
        .iter()
        .zip(shape)
        .map(|(&stride, &dimension)| {
            if stride < 0 || (dimension > 1 && stride == 0) {
                None
            } else {
                usize::try_from(stride).ok()
            }
        })
        .collect()
}

fn storage_span_len(shape: &[usize], strides: &[usize]) -> Option<usize> {
    if shape.iter().contains(&0) {
        return Some(0);
    }

    shape.iter().zip(strides).try_fold(1usize, |span, (&dimension, &stride)| {
        dimension.saturating_sub(1).checked_mul(stride).and_then(|extent| span.checked_add(extent))
    })
}

pub(crate) fn from_numpy_promoted<T>(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    source: DType,
) -> PyResult<NDArray<T>>
where
    T: ArrayElement + Element + RuntimeScalar,
{
    if source.name() == T::dtype().name() {
        return from_numpy(readonly_from_python::<T>(py, value)?);
    }

    crate::support::dtypes::with_dtype!(
        source,
        all | S | {
            from_numpy(readonly_from_python::<S>(py, value)?)?
                .astype_with_mode::<T>(atlas_ndarray::CastMode::Lossy)
                .map_err(|error| crate::support::errors::ndarray(py, error))
        }
    )
}

pub(crate) fn feature_matrix_f64(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<NDArray<f64>> {
    from_numpy(readonly_from_python(py, value)?)
}

pub(crate) fn label_vector_usize(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<NDArray<usize>> {
    from_numpy(readonly_from_python(py, value)?)
}

pub(crate) fn target_vector_f64(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<NDArray<f64>> {
    vector_f64(py, value)
}

pub(crate) fn vector_f64(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<NDArray<f64>> {
    from_numpy(readonly_from_python(py, value)?)
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

pub(crate) fn to_numpy_output<T>(
    py: Python<'_>,
    array: NDArray<T>,
    out: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>>
where
    T: ArrayElement + Element,
{
    let Some(out) = out else {
        return Ok(to_numpy_owned(py, array)?.into_any().unbind());
    };

    validate_dtype::<T>(py, out)?;
    let mut destination = out.extract::<PyReadwriteArrayDyn<'_, T>>()?;
    if destination.shape() != array.shape() {
        return Err(PyValueError::new_err(format!(
            "out has shape {:?}, expected {:?}",
            destination.shape(),
            array.shape(),
        )));
    }

    for (destination, &value) in destination.as_array_mut().iter_mut().zip(array.data()) {
        *destination = value;
    }

    Ok(out.clone().unbind())
}

pub(crate) fn to_numpy_f64_vector(py: Python<'_>, values: &[f64]) -> Py<PyAny> {
    PyArray1::from_slice(py, values).into_any().unbind()
}

pub(crate) fn to_numpy_usize_vector(py: Python<'_>, values: &[usize]) -> Py<PyAny> {
    PyArray1::from_slice(py, values).into_any().unbind()
}

pub(crate) fn metadata_dtype(py: Python<'_>, value: &Py<PyAny>) -> PyResult<Py<PyAny>> {
    value.bind(py).getattr("dtype").map(|value| value.unbind())
}

pub(crate) fn metadata_shape(py: Python<'_>, value: &Py<PyAny>) -> PyResult<Py<PyAny>> {
    value.bind(py).getattr("shape").map(|value| value.unbind())
}

pub(crate) fn metadata_ndim(py: Python<'_>, value: &Py<PyAny>) -> PyResult<usize> {
    value.bind(py).getattr("ndim")?.extract()
}

pub(crate) fn metadata_len(py: Python<'_>, value: &Py<PyAny>) -> PyResult<usize> {
    value.bind(py).len()
}

pub(crate) fn values_equal(py: Python<'_>, left: &Py<PyAny>, right: &Py<PyAny>) -> PyResult<bool> {
    PyModule::import(py, "numpy")?
        .getattr("array_equal")?
        .call1((left.bind(py), right.bind(py)))?
        .extract()
}

pub(crate) fn validate_dtype<T>(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()>
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
