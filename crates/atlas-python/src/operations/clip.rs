use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{array, gil};

pub(crate) fn clip(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    minimum: Option<&Bound<'_, PyAny>>,
    maximum: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    if minimum.is_none() && maximum.is_none() {
        return Err(crate::error::ndarray(
            py,
            atlas_ndarray::AtlasNdError::InvalidArgument {
                op: "clip",
                reason: "at least one bound is required",
            },
        ));
    }
    array::require_numpy_array(py, value)?;
    let dtype: String = value.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let value = array::from_numpy(array::readonly_from_python::<$ty>(py, value)?)?;
            let minimum = minimum.map(|minimum| minimum.extract::<$ty>()).transpose()?;
            let maximum = maximum.map(|maximum| maximum.extract::<$ty>()).transpose()?;
            let result = gil::without_gil(py, move || match (minimum, maximum) {
                (Some(minimum), Some(maximum)) => value.clip(minimum, maximum),
                (Some(minimum), None) => {
                    Ok(value.map(|element| if element < minimum { minimum } else { element }))
                }
                (None, Some(maximum)) => {
                    Ok(value.map(|element| if element > maximum { maximum } else { element }))
                }
                (None, None) => unreachable!("clip validates that at least one bound is present"),
            })
            .map_err(|error| crate::error::ndarray(py, error))?;
            Ok(array::to_numpy_owned(py, result)?.into_any().unbind())
        }};
    }

    match dtype.as_str() {
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
        "bool" => Err(PyTypeError::new_err("clip does not support bool dtype")),
        _ => Err(PyTypeError::new_err(format!("unsupported NumPy dtype {dtype}"))),
    }
}
