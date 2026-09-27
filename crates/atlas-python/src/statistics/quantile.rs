use atlas_stats::QuantileInterpolation;
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};

use crate::{array, gil};

#[derive(Clone, Copy)]
enum ScalarStatistic {
    Median,
    Quantile { q: f64, interpolation: QuantileInterpolation },
}

pub(crate) fn median(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<f64> {
    scalar_statistic(py, value, ScalarStatistic::Median)
}

pub(crate) fn quantile(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    q: f64,
    interpolation: &str,
) -> PyResult<f64> {
    scalar_statistic(
        py,
        value,
        ScalarStatistic::Quantile { q, interpolation: parse_interpolation(interpolation)? },
    )
}

fn scalar_statistic(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    statistic: ScalarStatistic,
) -> PyResult<f64> {
    array::require_numpy_array(py, value)?;
    let dtype: String = value.getattr("dtype")?.getattr("name")?.extract()?;

    macro_rules! apply {
        ($ty:ty) => {{
            let value = array::from_numpy(array::readonly_from_python::<$ty>(py, value)?)?;
            gil::without_gil(py, move || match statistic {
                ScalarStatistic::Median => atlas_stats::median(&value),
                ScalarStatistic::Quantile { q, interpolation } => {
                    atlas_stats::quantile_with_interpolation(&value, q, interpolation)
                }
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
        "bool" => Err(PyTypeError::new_err("quantile statistics do not support bool dtype")),
        _ => Err(PyTypeError::new_err(format!("unsupported NumPy dtype {dtype}"))),
    }
}

fn parse_interpolation(interpolation: &str) -> PyResult<QuantileInterpolation> {
    match interpolation {
        "linear" => Ok(QuantileInterpolation::Linear),
        "lower" => Ok(QuantileInterpolation::Lower),
        "higher" => Ok(QuantileInterpolation::Higher),
        "nearest" => Ok(QuantileInterpolation::Nearest),
        "midpoint" => Ok(QuantileInterpolation::Midpoint),
        _ => Err(PyValueError::new_err(
            "interpolation must be 'linear', 'lower', 'higher', 'nearest', or 'midpoint'",
        )),
    }
}
