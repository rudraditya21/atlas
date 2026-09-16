use atlas_ndarray::{AtlasNdError, AxisIndex, NDArray, Numeric, checked_element_count};
use num_traits::ToPrimitive;

use crate::{AtlasStatsError, AtlasStatsResult, StatsOperand};

#[derive(Default)]
struct RunningVariance {
    count: usize,
    mean: f64,
    sum_squares: f64,
}

impl RunningVariance {
    fn add(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        self.sum_squares += delta * (value - self.mean);
    }

    fn variance(self, ddof: usize) -> f64 {
        self.sum_squares / (self.count - ddof) as f64
    }
}

pub fn variance_axis<'a, T, I, A>(input: I, axis: A) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    I: Into<StatsOperand<'a, T>>,
    A: AxisIndex,
{
    variance_axis_impl(input.into(), axis, 0, false, "variance_axis")
}

pub fn variance_axis_ddof<'a, T, I, A>(
    input: I,
    axis: A,
    ddof: usize,
) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    I: Into<StatsOperand<'a, T>>,
    A: AxisIndex,
{
    variance_axis_impl(input.into(), axis, ddof, false, "variance_axis_ddof")
}

/// Returns population variances after reducing `axis`, retaining it with length one.
pub fn variance_axis_keepdims<'a, T, I, A>(input: I, axis: A) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    I: Into<StatsOperand<'a, T>>,
    A: AxisIndex,
{
    variance_axis_impl(input.into(), axis, 0, true, "variance_axis_keepdims")
}

/// Returns variances normalized by each lane length minus `ddof`, retaining `axis` with length one.
pub fn variance_axis_keepdims_ddof<'a, T, I, A>(
    input: I,
    axis: A,
    ddof: usize,
) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    I: Into<StatsOperand<'a, T>>,
    A: AxisIndex,
{
    variance_axis_impl(input.into(), axis, ddof, true, "variance_axis_keepdims_ddof")
}

pub(super) fn variance_axis_impl<T, A>(
    input: StatsOperand<'_, T>,
    axis: A,
    ddof: usize,
    keepdims: bool,
    op: &'static str,
) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    A: AxisIndex,
{
    let shape = input.shape();
    let axis = normalize_axis(axis, shape.len())?;
    let output_shape = axis_output_shape(shape, axis, keepdims);
    let output_len = checked_element_count(&output_shape)?;
    if output_len == 0 {
        return NDArray::from_shape_vec(output_shape, Vec::new()).map_err(Into::into);
    }

    let axis_len = shape[axis];
    if axis_len == 0 {
        return Err(AtlasStatsError::EmptyInput { op });
    }
    if ddof >= axis_len {
        return Err(AtlasStatsError::InvalidDegreesOfFreedom { op, ddof, count: axis_len });
    }

    let mut variances = (0..output_len).map(|_| RunningVariance::default()).collect::<Vec<_>>();

    try_for_each_axis_value(&input, axis, |lane, value| -> AtlasStatsResult<()> {
        variances[lane].add(value.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op })?);
        Ok(())
    })?;

    NDArray::from_shape_vec(
        output_shape,
        variances.into_iter().map(|variance| variance.variance(ddof)).collect(),
    )
    .map_err(Into::into)
}

pub(super) fn axis_output_shape(shape: &[usize], axis: usize, keepdims: bool) -> Vec<usize> {
    if keepdims {
        let mut output_shape = shape.to_vec();
        output_shape[axis] = 1;
        output_shape
    } else {
        let mut output_shape = shape.to_vec();
        output_shape.remove(axis);
        output_shape
    }
}

pub(super) fn try_for_each_axis_value<T, F>(
    input: &StatsOperand<'_, T>,
    axis: usize,
    mut f: F,
) -> AtlasStatsResult<()>
where
    T: Numeric,
    F: FnMut(usize, &T) -> AtlasStatsResult<()>,
{
    let shape = input.shape();
    let axis_len = shape[axis];
    let inner_len = shape[axis + 1..].iter().product::<usize>();
    let block_len = axis_len * inner_len;

    if let Some(values) = input.row_major_slice() {
        if axis + 1 == shape.len() {
            for (lane, values) in values.chunks_exact(axis_len).enumerate() {
                for value in values {
                    f(lane, value)?;
                }
            }
            return Ok(());
        }

        if axis == 0 {
            for values in values.chunks_exact(inner_len) {
                for (lane, value) in values.iter().enumerate() {
                    f(lane, value)?;
                }
            }
            return Ok(());
        }

        for (linear_index, value) in values.iter().enumerate() {
            f(axis_lane(linear_index, inner_len, block_len), value)?;
        }
        return Ok(());
    }

    for (linear_index, value) in input.iter().enumerate() {
        f(axis_lane(linear_index, inner_len, block_len), value)?;
    }
    Ok(())
}

pub(super) fn try_for_each_axis_pair<T, F>(
    lhs: &StatsOperand<'_, T>,
    rhs: &StatsOperand<'_, T>,
    axis: usize,
    mut f: F,
) -> AtlasStatsResult<()>
where
    T: Numeric,
    F: FnMut(usize, &T, &T) -> AtlasStatsResult<()>,
{
    let shape = lhs.shape();
    let axis_len = shape[axis];
    let inner_len = shape[axis + 1..].iter().product::<usize>();
    let block_len = axis_len * inner_len;

    if let (Some(lhs_values), Some(rhs_values)) = (lhs.row_major_slice(), rhs.row_major_slice()) {
        if axis + 1 == shape.len() {
            for (lane, (lhs_lane, rhs_lane)) in
                lhs_values.chunks_exact(axis_len).zip(rhs_values.chunks_exact(axis_len)).enumerate()
            {
                for (lhs_value, rhs_value) in lhs_lane.iter().zip(rhs_lane) {
                    f(lane, lhs_value, rhs_value)?;
                }
            }
            return Ok(());
        }

        if axis == 0 {
            for (lhs_values, rhs_values) in
                lhs_values.chunks_exact(inner_len).zip(rhs_values.chunks_exact(inner_len))
            {
                for (lane, (lhs_value, rhs_value)) in lhs_values.iter().zip(rhs_values).enumerate()
                {
                    f(lane, lhs_value, rhs_value)?;
                }
            }
            return Ok(());
        }

        for (linear_index, (lhs_value, rhs_value)) in lhs_values.iter().zip(rhs_values).enumerate()
        {
            f(axis_lane(linear_index, inner_len, block_len), lhs_value, rhs_value)?;
        }
        return Ok(());
    }

    for (linear_index, (lhs_value, rhs_value)) in lhs.iter().zip(rhs.iter()).enumerate() {
        f(axis_lane(linear_index, inner_len, block_len), lhs_value, rhs_value)?;
    }
    Ok(())
}

fn axis_lane(linear_index: usize, inner_len: usize, block_len: usize) -> usize {
    linear_index / block_len * inner_len + linear_index % inner_len
}

pub fn stddev_axis<'a, T, I, A>(input: I, axis: A) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    I: Into<StatsOperand<'a, T>>,
    A: AxisIndex,
{
    stddev_axis_impl(input.into(), axis, 0, false, "stddev_axis")
}

/// Returns population standard deviations after reducing `axis`, retaining it with length one.
pub fn stddev_axis_keepdims<'a, T, I, A>(input: I, axis: A) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    I: Into<StatsOperand<'a, T>>,
    A: AxisIndex,
{
    stddev_axis_impl(input.into(), axis, 0, true, "stddev_axis_keepdims")
}

/// Returns standard deviations normalized by each lane length minus `ddof`, retaining `axis` with length one.
pub fn stddev_axis_keepdims_ddof<'a, T, I, A>(
    input: I,
    axis: A,
    ddof: usize,
) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    I: Into<StatsOperand<'a, T>>,
    A: AxisIndex,
{
    stddev_axis_impl(input.into(), axis, ddof, true, "stddev_axis_keepdims_ddof")
}

fn stddev_axis_impl<T, A>(
    input: StatsOperand<'_, T>,
    axis: A,
    ddof: usize,
    keepdims: bool,
    op: &'static str,
) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    A: AxisIndex,
{
    let values = variance_axis_impl(input, axis, ddof, keepdims, op)?;
    NDArray::from_shape_vec(
        values.shape().to_vec(),
        values.data().iter().map(|value| value.sqrt()).collect(),
    )
    .map_err(Into::into)
}

pub fn stddev_axis_ddof<'a, T, I, A>(
    input: I,
    axis: A,
    ddof: usize,
) -> AtlasStatsResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
    I: Into<StatsOperand<'a, T>>,
    A: AxisIndex,
{
    stddev_axis_impl(input.into(), axis, ddof, false, "stddev_axis_ddof")
}

pub(crate) fn normalize_axis<A: AxisIndex>(axis: A, ndim: usize) -> AtlasStatsResult<usize> {
    let axis = axis.try_into_i64().ok_or(AtlasNdError::InvalidAxis { axis: i64::MAX, ndim })?;
    let normalized = if axis < 0 { ndim as i64 + axis } else { axis };

    if normalized < 0 || normalized >= ndim as i64 {
        return Err(AtlasNdError::InvalidAxis { axis, ndim }.into());
    }

    Ok(normalized as usize)
}
