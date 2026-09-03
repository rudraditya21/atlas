use atlas_ndarray::Numeric;
use num_traits::ToPrimitive;

use crate::{AtlasStatsError, AtlasStatsResult, StatsOperand};

/// Returns the linearly interpolated quantile at `q` for a rank-1 or higher-dimensional operand.
///
/// Values are flattened in logical row-major order. Any input NaN propagates to the result.
pub fn quantile<'a, T, I>(input: I, q: f64) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive,
    I: Into<StatsOperand<'a, T>>,
{
    if !q.is_finite() || !(0.0..=1.0).contains(&q) {
        return Err(AtlasStatsError::InvalidQuantile {
            reason: "must be finite and within [0, 1]",
        });
    }

    let input = input.into();
    if input.len()? == 0 {
        return Err(AtlasStatsError::EmptyInput { op: "quantile" });
    }

    let mut values = Vec::with_capacity(input.len()?);
    for value in input.iter() {
        let value =
            value.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op: "quantile" })?;
        if value.is_nan() {
            return Ok(f64::NAN);
        }
        values.push(value);
    }
    values.sort_unstable_by(f64::total_cmp);

    let rank = q * (values.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    Ok(values[lower] + (values[upper] - values[lower]) * (rank - lower as f64))
}

/// Returns the median using the same linear interpolation and NaN policy as [`quantile`].
pub fn median<'a, T, I>(input: I) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive,
    I: Into<StatsOperand<'a, T>>,
{
    quantile(input, 0.5).map_err(|error| match error {
        AtlasStatsError::EmptyInput { .. } => AtlasStatsError::EmptyInput { op: "median" },
        AtlasStatsError::NumericConversionFailed { .. } => {
            AtlasStatsError::NumericConversionFailed { op: "median" }
        }
        error => error,
    })
}
