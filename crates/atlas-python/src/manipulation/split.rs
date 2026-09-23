use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{array, gil};

enum SplitSpec {
    Indices(Vec<usize>),
    Sections(usize),
}

pub(crate) fn split(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    indices_or_sections: &Bound<'_, PyAny>,
    axis: i64,
) -> PyResult<Vec<Py<PyAny>>> {
    let split_spec = split_spec(indices_or_sections)?;
    array::require_numpy_array(py, value)?;
    let dtype: String = value.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let array = array::from_numpy(array::readonly_from_python::<$ty>(py, value)?)?;
            let arrays = gil::without_gil(py, move || {
                let views = match split_spec {
                    SplitSpec::Indices(indices) => array.split_at_indices(&indices, axis),
                    SplitSpec::Sections(sections) => array.split(sections, axis),
                };
                views.map(|views| views.into_iter().map(|view| view.to_owned()).collect::<Vec<_>>())
            })
            .map_err(|error| crate::error::ndarray(py, error))?;
            arrays
                .into_iter()
                .map(|array| Ok(array::to_numpy_owned(py, array)?.into_any().unbind()))
                .collect()
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

fn split_spec(value: &Bound<'_, PyAny>) -> PyResult<SplitSpec> {
    value
        .extract::<usize>()
        .map(SplitSpec::Sections)
        .or_else(|_| value.extract::<Vec<usize>>().map(SplitSpec::Indices))
}
