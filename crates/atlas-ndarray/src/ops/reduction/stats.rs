use num_traits::ToPrimitive;

use super::metadata::{AxisReductionMetadata, ReductionOperand, WholeReductionMetadata};
use crate::{
    AtlasNdError, AtlasNdResult, AxisIndex, NDArray, Numeric, OperandMetadata,
    internal::{
        for_each_value,
        layout::{LayoutKind, dense_storage_slice},
        offset_iter, simd,
    },
};

#[derive(Default)]
pub(super) struct RunningVariance {
    count: usize,
    mean: f64,
    sum_squares: f64,
}

impl RunningVariance {
    pub(super) fn add(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        self.sum_squares += delta * (value - self.mean);
    }

    pub(super) fn population(&self) -> f64 {
        self.sum_squares / self.count as f64
    }
}

pub(super) fn variance_all<T: Numeric + ToPrimitive>(
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    op: &'static str,
) -> AtlasNdResult<f64> {
    let metadata = WholeReductionMetadata::from_shape(shape);
    metadata.require_non_empty(op)?;

    if let Some(values) = dense_storage_slice(data, offset, shape, strides) {
        return variance_contiguous(values, op);
    }

    let mut variance = RunningVariance::default();
    try_add_all(&mut variance, data, offset, shape, strides, op)?;
    Ok(variance.population())
}

pub(super) fn variance_axis<T: Numeric + ToPrimitive, O: OperandMetadata<T> + ?Sized>(
    operand: &O,
    axis: impl AxisIndex,
    keepdims: bool,
    stddev: bool,
    op: &'static str,
) -> AtlasNdResult<NDArray<f64>> {
    variance_axis_impl(ReductionOperand::new(operand), axis, keepdims, stddev, op)
}

fn variance_axis_impl<T: Numeric + ToPrimitive>(
    operand: ReductionOperand<'_, T>,
    axis: impl AxisIndex,
    keepdims: bool,
    stddev: bool,
    op: &'static str,
) -> AtlasNdResult<NDArray<f64>> {
    let metadata = AxisReductionMetadata::new(operand.shape, operand.strides, axis, keepdims)?;
    metadata.require_non_empty(op)?;

    if metadata.axis_layout == LayoutKind::Contiguous
        && (simd::is_f32::<T>() || simd::is_f64::<T>())
    {
        if metadata.source_layout == LayoutKind::Contiguous {
            let source = &operand.data[operand.offset
                ..operand.offset + metadata.output.len.saturating_mul(metadata.axis_len)];
            let mut values = Vec::with_capacity(metadata.output.len);
            for lane in source.chunks_exact(metadata.axis_len) {
                let variance = variance_contiguous(lane, op)?;
                values.push(if stddev { variance.sqrt() } else { variance });
            }
            return NDArray::from_shape_vec(metadata.output.shape, values);
        }

        let mut values = Vec::with_capacity(metadata.output.len);
        for lane_offset in
            offset_iter(operand.offset, &metadata.output.shape, &metadata.output.outer_strides)
        {
            let variance = variance_contiguous(
                &operand.data[lane_offset..lane_offset + metadata.axis_len],
                op,
            )?;
            values.push(if stddev { variance.sqrt() } else { variance });
        }
        return NDArray::from_shape_vec(metadata.output.shape, values);
    }

    let mut values = Vec::with_capacity(metadata.output.len);

    for lane_offset in
        offset_iter(operand.offset, &metadata.output.shape, &metadata.output.outer_strides)
    {
        let mut variance = RunningVariance::default();
        for index in 0..metadata.axis_len {
            variance.add(
                operand.data[lane_offset + index * metadata.axis_stride]
                    .to_f64()
                    .ok_or(AtlasNdError::NumericConversionFailed { op })?,
            );
        }
        let value = variance.population();
        values.push(if stddev { value.sqrt() } else { value });
    }

    NDArray::from_shape_vec(metadata.output.shape, values)
}

fn variance_contiguous<T: Numeric + ToPrimitive>(
    values: &[T],
    op: &'static str,
) -> AtlasNdResult<f64> {
    if simd::is_f32::<T>() {
        let values = simd::cast_slice::<T, f32>(values);
        let mean = simd::compensated_sum_f32_as_f64(values) / values.len() as f64;
        return Ok(simd::squared_deviations_f32(values, mean) / values.len() as f64);
    }

    if simd::is_f64::<T>() {
        let values = simd::cast_slice::<T, f64>(values);
        let mean = simd::compensated_sum_f64(values) / values.len() as f64;
        return Ok(simd::squared_deviations_f64(values, mean) / values.len() as f64);
    }

    let mut variance = RunningVariance::default();
    for value in values {
        variance.add(value.to_f64().ok_or(AtlasNdError::NumericConversionFailed { op })?);
    }
    Ok(variance.population())
}

fn try_add_all<T: ToPrimitive>(
    variance: &mut RunningVariance,
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    op: &'static str,
) -> AtlasNdResult<()> {
    let mut error = None;
    for_each_value(data, offset, shape, strides, |value| match value.to_f64() {
        Some(value) => variance.add(value),
        None => error = Some(AtlasNdError::NumericConversionFailed { op }),
    });
    error.map_or(Ok(()), Err)
}
