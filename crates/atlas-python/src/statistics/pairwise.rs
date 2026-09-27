use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{array, gil};

#[derive(Clone, Copy)]
enum Statistic {
    Covariance,
    Correlation,
}

pub(crate) fn covariance(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    pairwise_statistic(py, lhs, rhs, Statistic::Covariance)
}

pub(crate) fn correlation(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    pairwise_statistic(py, lhs, rhs, Statistic::Correlation)
}

fn pairwise_statistic(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    rhs: &Bound<'_, PyAny>,
    statistic: Statistic,
) -> PyResult<f64> {
    array::require_numpy_array(py, lhs)?;
    let dtype: String = lhs.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let lhs = array::from_numpy(array::readonly_from_python::<$ty>(py, lhs)?)?;
            let rhs = array::from_numpy(array::readonly_from_python::<$ty>(py, rhs)?)?;
            gil::without_gil(py, move || match statistic {
                Statistic::Covariance => atlas_stats::covariance(&lhs, &rhs),
                Statistic::Correlation => atlas_stats::correlation(&lhs, &rhs),
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
        "bool" => Err(PyTypeError::new_err("pairwise statistics do not support bool dtype")),
        _ => Err(PyTypeError::new_err(format!("unsupported NumPy dtype {dtype}"))),
    }
}
