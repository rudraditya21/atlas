use num_traits::ToPrimitive;
use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{array, gil};

pub(crate) fn slogdet(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<(f64, f64)> {
    array::require_numpy_array(py, value)?;
    let dtype: String = value.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let value = array::from_numpy(array::readonly_from_python::<$ty>(py, value)?)?;
            gil::without_gil(py, move || atlas_linalg::slogdet(&value))
                .map(|(sign, log_abs_det)| {
                    (
                        sign.to_f64().expect("floating-point sign must convert to f64"),
                        log_abs_det
                            .to_f64()
                            .expect("floating-point log determinant must convert to f64"),
                    )
                })
                .map_err(|error| crate::error::linalg(py, error))
        }};
    }

    match dtype.as_str() {
        "float32" => apply!(f32),
        "float64" => apply!(f64),
        _ => Err(PyTypeError::new_err("slogdet requires a float32 or float64 NumPy array")),
    }
}
