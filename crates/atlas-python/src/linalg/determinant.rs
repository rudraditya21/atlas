use atlas_ndarray::{AtlasNdError, NDArray};
use num_traits::ToPrimitive;
use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{array, gil};

pub(crate) fn det(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<f64> {
    array::require_numpy_array(py, value)?;
    let dtype: String = value.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let value = array::from_numpy(array::readonly_from_python::<$ty>(py, value)?)?;
            gil::without_gil(py, move || atlas_linalg::det(&to_f64_array(value)?))
                .map_err(|error| crate::error::linalg(py, error))
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
        "bool" => Err(PyTypeError::new_err("det does not support bool dtype")),
        _ => Err(PyTypeError::new_err(format!("unsupported NumPy dtype {dtype}"))),
    }
}

fn to_f64_array<T>(array: NDArray<T>) -> atlas_linalg::AtlasLinalgResult<NDArray<f64>>
where
    T: ToPrimitive + atlas_ndarray::ArrayElement,
{
    let values = array
        .data()
        .iter()
        .map(|value| value.to_f64().ok_or(AtlasNdError::NumericConversionFailed { op: "det" }))
        .collect::<Result<Vec<_>, _>>()?;

    NDArray::from_shape_vec(array.shape().to_vec(), values).map_err(Into::into)
}
