use atlas_ndarray::AtlasNdError;
use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{array, gil};

pub(crate) fn pad(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    widths: &Bound<'_, PyAny>,
    fill: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    array::require_numpy_array(py, value)?;
    let ndim = value.getattr("ndim")?.extract()?;
    let widths = pad_widths(py, widths, ndim)?;
    let dtype: String = value.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let array = array::from_numpy(array::readonly_from_python::<$ty>(py, value)?)?;
            let fill = fill.map(|fill| fill.extract::<$ty>()).transpose()?.unwrap_or_default();
            let result = gil::without_gil(py, move || array.pad(&widths, fill))
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

fn pad_widths(
    py: Python<'_>,
    widths: &Bound<'_, PyAny>,
    ndim: usize,
) -> PyResult<Vec<(usize, usize)>> {
    let widths = if let Ok(width) = widths.extract::<i64>() {
        vec![(width, width); ndim]
    } else if let Ok(pair) = widths.extract::<(i64, i64)>() {
        vec![pair; ndim]
    } else {
        widths.extract::<Vec<(i64, i64)>>()?
    };

    widths
        .into_iter()
        .map(|(before, after)| {
            Ok((
                usize::try_from(before).map_err(|_| invalid_width(py))?,
                usize::try_from(after).map_err(|_| invalid_width(py))?,
            ))
        })
        .collect()
}

fn invalid_width(py: Python<'_>) -> PyErr {
    crate::error::ndarray(
        py,
        AtlasNdError::InvalidArgument { op: "pad", reason: "widths must be nonnegative" },
    )
}
