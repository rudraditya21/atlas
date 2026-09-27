use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{array, gil};

#[derive(Clone, Copy)]
enum Statistic {
    Mean,
    Variance,
}

pub(crate) fn weighted_mean(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
    weights: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    weighted_statistic(py, values, weights, Statistic::Mean)
}

pub(crate) fn weighted_variance(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
    weights: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    weighted_statistic(py, values, weights, Statistic::Variance)
}

fn weighted_statistic(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
    weights: &Bound<'_, PyAny>,
    statistic: Statistic,
) -> PyResult<f64> {
    array::require_numpy_array(py, values)?;
    let dtype: String = values.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let values = array::from_numpy(array::readonly_from_python::<$ty>(py, values)?)?;
            let weights = array::from_numpy(array::readonly_from_python::<$ty>(py, weights)?)?;
            gil::without_gil(py, move || match statistic {
                Statistic::Mean => atlas_stats::weighted_mean(&values, &weights),
                Statistic::Variance => atlas_stats::weighted_variance(&values, &weights),
            })
            .map_err(|error| crate::error::stats(py, error))
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
        "bool" => Err(PyTypeError::new_err("weighted statistics do not support bool dtype")),
        _ => Err(PyTypeError::new_err(format!("unsupported NumPy dtype {dtype}"))),
    }
}
