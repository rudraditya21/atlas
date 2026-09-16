use atlas_ndarray::{AtlasNdError, AxisIndex, NDArray, Numeric, checked_element_count};
use num_traits::ToPrimitive;

use crate::{AtlasStatsError, AtlasStatsResult, StatsOperand};

/// Interpolation used when a quantile rank lies between two sorted values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QuantileInterpolation {
    /// Linearly interpolate between the surrounding values.
    #[default]
    Linear,
    /// Select the lower surrounding value.
    Lower,
    /// Select the higher surrounding value.
    Higher,
    /// Select the nearest surrounding value, resolving midpoint ties upward.
    Nearest,
    /// Return the arithmetic mean of the surrounding values.
    Midpoint,
}

/// Returns the linearly interpolated quantile at `q` for a rank-1 or higher-dimensional operand.
///
/// Values are flattened in logical row-major order. Any input NaN propagates to the result.
pub fn quantile<'a, T, I>(input: I, q: f64) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive,
    I: Into<StatsOperand<'a, T>>,
{
    quantile_with_interpolation(input, q, QuantileInterpolation::Linear)
}

/// Returns the quantile at `q` using `interpolation` between surrounding sorted values.
pub fn quantile_with_interpolation<'a, T, I>(
    input: I,
    q: f64,
    interpolation: QuantileInterpolation,
) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive,
    I: Into<StatsOperand<'a, T>>,
{
    validate_quantile(q)?;

    let input = input.into();
    if input.len()? == 0 {
        return Err(AtlasStatsError::EmptyInput { op: "quantile" });
    }

    let mut values = Vec::with_capacity(input.len()?);
    let mut contains_nan = false;
    input.try_for_each_span(|span| -> AtlasStatsResult<()> {
        if contains_nan {
            return Ok(());
        }

        for value in span {
            let value = value
                .to_f64()
                .ok_or(AtlasStatsError::NumericConversionFailed { op: "quantile" })?;
            if value.is_nan() {
                contains_nan = true;
                break;
            }
            values.push(value);
        }

        Ok(())
    })?;
    if contains_nan {
        return Ok(f64::NAN);
    }
    values.sort_unstable_by(f64::total_cmp);

    Ok(interpolate_quantile(&values, q, interpolation))
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
    quantile_axis_with_interpolation(input, q, axis, QuantileInterpolation::Linear)
}

/// Returns quantiles after reducing `axis` using `interpolation` between surrounding sorted values.
pub fn quantile_axis_with_interpolation<'a, T, I, A>(
    input: I,
    q: f64,
    axis: A,
    interpolation: QuantileInterpolation,
) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    I: Into<StatsOperand<'a, T>>,
    A: AxisIndex,
{
    quantile_axis_impl(input.into(), q, axis, interpolation, false, "quantile_axis")
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
    quantile_axis_keepdims_with_interpolation(input, q, axis, QuantileInterpolation::Linear)
}

/// Returns quantiles after reducing `axis`, retaining it with length one.
pub fn quantile_axis_keepdims_with_interpolation<'a, T, I, A>(
    input: I,
    q: f64,
    axis: A,
    interpolation: QuantileInterpolation,
) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    I: Into<StatsOperand<'a, T>>,
    A: AxisIndex,
{
    quantile_axis_impl(input.into(), q, axis, interpolation, true, "quantile_axis")
}

/// Returns medians after reducing `axis`.
pub fn median_axis<'a, T, I, A>(input: I, axis: A) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    I: Into<StatsOperand<'a, T>>,
    A: AxisIndex,
{
    quantile_axis_impl(input.into(), 0.5, axis, QuantileInterpolation::Linear, false, "median_axis")
}

/// Returns medians after reducing `axis`, retaining it with length one.
pub fn median_axis_keepdims<'a, T, I, A>(input: I, axis: A) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    I: Into<StatsOperand<'a, T>>,
    A: AxisIndex,
{
    quantile_axis_impl(input.into(), 0.5, axis, QuantileInterpolation::Linear, true, "median_axis")
}

fn quantile_axis_impl<T, A>(
    input: StatsOperand<'_, T>,
    q: f64,
    axis: A,
    interpolation: QuantileInterpolation,
    keepdims: bool,
    op: &'static str,
) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    A: AxisIndex,
{
    validate_quantile(q)?;
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
        result.push(interpolate_quantile(&lane, q, interpolation));
    }
    if keepdims {
        output_shape.insert(axis, 1);
    }
    NDArray::from_shape_vec(output_shape, result).map_err(Into::into)
}

fn validate_quantile(q: f64) -> AtlasStatsResult<()> {
    if !q.is_finite() || !(0.0..=1.0).contains(&q) {
        return Err(AtlasStatsError::InvalidQuantile {
            reason: "must be finite and within [0, 1]",
        });
    }

    Ok(())
}

fn interpolate_quantile(values: &[f64], q: f64, interpolation: QuantileInterpolation) -> f64 {
    let rank = q * (values.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;

    match interpolation {
        QuantileInterpolation::Linear => {
            values[lower] + (values[upper] - values[lower]) * (rank - lower as f64)
        }
        QuantileInterpolation::Lower => values[lower],
        QuantileInterpolation::Higher => values[upper],
        QuantileInterpolation::Nearest => values[rank.round() as usize],
        QuantileInterpolation::Midpoint => (values[lower] + values[upper]) / 2.0,
    }
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use crate::{
        QuantileInterpolation, quantile, quantile_axis_with_interpolation,
        quantile_with_interpolation,
    };

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() <= 1e-12);
    }

    #[test]
    fn quantile_interpolation_preserves_exact_ranks() {
        let values = NDArray::from_shape_vec([4], vec![0.0_f64, 10.0, 20.0, 30.0]).unwrap();

        for interpolation in [
            QuantileInterpolation::Linear,
            QuantileInterpolation::Lower,
            QuantileInterpolation::Higher,
            QuantileInterpolation::Nearest,
            QuantileInterpolation::Midpoint,
        ] {
            assert_close(
                quantile_with_interpolation(&values, 2.0 / 3.0, interpolation).unwrap(),
                20.0,
            );
        }
    }

    #[test]
    fn quantile_interpolation_handles_even_lengths() {
        let values = NDArray::from_shape_vec([4], vec![0.0_f64, 10.0, 20.0, 30.0]).unwrap();

        assert_close(quantile(&values, 0.5).unwrap(), 15.0);
        assert_close(
            quantile_with_interpolation(&values, 0.5, QuantileInterpolation::Lower).unwrap(),
            10.0,
        );
        assert_close(
            quantile_with_interpolation(&values, 0.5, QuantileInterpolation::Higher).unwrap(),
            20.0,
        );
        assert_close(
            quantile_with_interpolation(&values, 0.5, QuantileInterpolation::Nearest).unwrap(),
            20.0,
        );
        assert_close(
            quantile_with_interpolation(&values, 0.5, QuantileInterpolation::Midpoint).unwrap(),
            15.0,
        );
    }

    #[test]
    fn quantile_interpolation_preserves_edge_quantiles() {
        let values = NDArray::from_shape_vec([4], vec![30.0_f64, 0.0, 20.0, 10.0]).unwrap();

        assert_close(
            quantile_with_interpolation(&values, 0.0, QuantileInterpolation::Higher).unwrap(),
            0.0,
        );
        assert_close(
            quantile_with_interpolation(&values, 1.0, QuantileInterpolation::Lower).unwrap(),
            30.0,
        );
    }

    #[test]
    fn quantile_interpolation_supports_axis_reductions_and_views() {
        let values = NDArray::from_shape_vec(
            [2, 4],
            vec![0.0_f64, 10.0, 20.0, 30.0, 100.0, 110.0, 120.0, 130.0],
        )
        .unwrap();

        assert_eq!(
            quantile_axis_with_interpolation(&values, 0.25, 1, QuantileInterpolation::Lower)
                .unwrap()
                .data(),
            &[0.0, 100.0]
        );
        assert_eq!(
            quantile_axis_with_interpolation(
                values.view().transpose(),
                0.25,
                0,
                QuantileInterpolation::Midpoint,
            )
            .unwrap()
            .data(),
            &[5.0, 105.0]
        );
    }
}
