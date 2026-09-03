use atlas_ndarray::{AtlasNdError, AxisIndex, NDArray, Numeric, checked_element_count};
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

/// Returns linearly interpolated quantiles after reducing `axis`.
pub fn quantile_axis<'a, T, I, A>(input: I, q: f64, axis: A) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    I: Into<StatsOperand<'a, T>>,
    A: AxisIndex,
{
    quantile_axis_impl(input.into(), q, axis, false, "quantile_axis")
}

/// Returns linearly interpolated quantiles after reducing `axis`, retaining it with length one.
pub fn quantile_axis_keepdims<'a, T, I, A>(
    input: I,
    q: f64,
    axis: A,
) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    I: Into<StatsOperand<'a, T>>,
    A: AxisIndex,
{
    quantile_axis_impl(input.into(), q, axis, true, "quantile_axis")
}

/// Returns medians after reducing `axis`.
pub fn median_axis<'a, T, I, A>(input: I, axis: A) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    I: Into<StatsOperand<'a, T>>,
    A: AxisIndex,
{
    quantile_axis_impl(input.into(), 0.5, axis, false, "median_axis")
}

/// Returns medians after reducing `axis`, retaining it with length one.
pub fn median_axis_keepdims<'a, T, I, A>(input: I, axis: A) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    I: Into<StatsOperand<'a, T>>,
    A: AxisIndex,
{
    quantile_axis_impl(input.into(), 0.5, axis, true, "median_axis")
}

fn quantile_axis_impl<T, A>(
    input: StatsOperand<'_, T>,
    q: f64,
    axis: A,
    keepdims: bool,
    op: &'static str,
) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    A: AxisIndex,
{
    if !q.is_finite() || !(0.0..=1.0).contains(&q) {
        return Err(AtlasStatsError::InvalidQuantile {
            reason: "must be finite and within [0, 1]",
        });
    }
    let shape = input.shape();
    let raw_axis = axis
        .try_into_i64()
        .ok_or(AtlasNdError::InvalidAxis { axis: i64::MAX, ndim: shape.len() })?;
    let axis = if raw_axis < 0 { shape.len() as i64 + raw_axis } else { raw_axis };
    if axis < 0 || axis >= shape.len() as i64 {
        return Err(AtlasNdError::InvalidAxis { axis: raw_axis, ndim: shape.len() }.into());
    }
    let axis = axis as usize;
    let axis_len = shape[axis];
    let mut output_shape = shape.to_vec();
    output_shape.remove(axis);
    let output_len = checked_element_count(&output_shape)?;
    if output_len == 0 {
        if keepdims {
            output_shape.insert(axis, 1);
        }
        return NDArray::from_shape_vec(output_shape, Vec::new()).map_err(Into::into);
    }
    if axis_len == 0 {
        return Err(AtlasStatsError::EmptyInput { op });
    }
    let inner = shape[axis + 1..].iter().product::<usize>();
    let block = axis_len * inner;
    let mut lanes = vec![Vec::with_capacity(axis_len); output_len];
    for (index, value) in input.iter().enumerate() {
        lanes[(index / block) * inner + index % inner]
            .push(value.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op })?);
    }
    let mut result = Vec::with_capacity(output_len);
    for mut lane in lanes {
        if lane.iter().any(|value| value.is_nan()) {
            result.push(f64::NAN);
            continue;
        }
        lane.sort_unstable_by(f64::total_cmp);
        let rank = q * (lane.len() - 1) as f64;
        let lower = rank.floor() as usize;
        let upper = rank.ceil() as usize;
        result.push(lane[lower] + (lane[upper] - lane[lower]) * (rank - lower as f64));
    }
    if keepdims {
        output_shape.insert(axis, 1);
    }
    NDArray::from_shape_vec(output_shape, result).map_err(Into::into)
}
