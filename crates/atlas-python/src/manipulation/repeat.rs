use atlas_ndarray::AtlasNdError;
use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{array, gil};

enum RepeatSpec {
    Scalar(usize),
    Counts(Vec<usize>),
}

pub(crate) fn repeat(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    repeats: &Bound<'_, PyAny>,
    axis: Option<i64>,
) -> PyResult<Py<PyAny>> {
    let repeats = repeat_spec(py, repeats)?;
    array::require_numpy_array(py, value)?;
    let dtype: String = value.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let array = array::from_numpy(array::readonly_from_python::<$ty>(py, value)?)?;
            let result = match repeats {
                RepeatSpec::Scalar(repeats) => gil::without_gil(py, move || match axis {
                    Some(axis) => array.repeat(repeats, axis),
                    None => array.flatten().repeat(repeats, 0),
                }),
                RepeatSpec::Counts(counts) => {
                    let indices = repeat_indices(&array, &counts, axis)
                        .map_err(|error| crate::error::ndarray(py, error))?;
                    gil::without_gil(py, move || match axis {
                        Some(axis) => array.take(&indices, axis),
                        None => array.flatten().take(&indices, 0),
                    })
                }
            }
            .map_err(|error| crate::error::ndarray(py, error))?;
            Ok(array::to_numpy_owned(py, result)?.into_any().unbind())
        }};
    }

    match dtype.as_str() {
        "bool" => apply!(bool),
        "int8" => apply!(i8),
        "int16" => apply!(i16),
        "int32" => apply!(i32),
        "int64" => apply!(i64),
        "uint8" => apply!(u8),
        "uint16" => apply!(u16),
        "uint32" => apply!(u32),
        "uint64" => apply!(u64),
        "float32" => apply!(f32),
        "float64" => apply!(f64),
        _ => Err(PyTypeError::new_err(format!("unsupported NumPy dtype {dtype}"))),
    }
}

fn repeat_spec(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<RepeatSpec> {
    match value.extract::<i64>() {
        Ok(repeats) => Ok(RepeatSpec::Scalar(repeat_count(py, repeats)?)),
        Err(_) => value
            .extract::<Vec<i64>>()?
            .into_iter()
            .map(|repeats| repeat_count(py, repeats))
            .collect::<PyResult<Vec<_>>>()
            .map(RepeatSpec::Counts),
    }
}

fn repeat_count(py: Python<'_>, repeats: i64) -> PyResult<usize> {
    usize::try_from(repeats).map_err(|_| {
        crate::error::ndarray(
            py,
            AtlasNdError::InvalidArgument { op: "repeat", reason: "repeats must be nonnegative" },
        )
    })
}

fn repeat_indices<T>(
    array: &atlas_ndarray::NDArray<T>,
    counts: &[usize],
    axis: Option<i64>,
) -> Result<Vec<i64>, AtlasNdError>
where
    T: atlas_ndarray::ArrayElement,
{
    let axis_length = match axis {
        Some(axis) => {
            let normalized = if axis < 0 { axis + array.ndim() as i64 } else { axis };
            if !(0..array.ndim() as i64).contains(&normalized) {
                return Err(AtlasNdError::InvalidAxis { axis, ndim: array.ndim() });
            }
            array.shape()[usize::try_from(normalized).expect("normalized axes are nonnegative")]
        }
        None => array.len(),
    };
    if counts.len() != axis_length {
        return Err(AtlasNdError::InvalidArgument {
            op: "repeat",
            reason: "repeat counts must match the selected axis length",
        });
    }

    let output_len = counts.iter().try_fold(0_usize, |total, &count| {
        total
            .checked_add(count)
            .ok_or(AtlasNdError::ShapeOverflow { op: "repeat", shape: counts.to_vec() })
    })?;
    let mut indices = Vec::with_capacity(output_len);
    for (index, &count) in counts.iter().enumerate() {
        let index = i64::try_from(index).expect("ndarray dimensions fit i64");
        indices.extend(std::iter::repeat_n(index, count));
    }

    Ok(indices)
}
