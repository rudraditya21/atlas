use num_traits::ToPrimitive;

use crate::{
    AtlasNdError, AtlasNdResult, AxisIndex, NDArray, Numeric, OperandMetadata,
    internal::{for_each_value, offset_iter},
};

use super::metadata::{AxisReductionMetadata, WholeReductionMetadata};

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
    WholeReductionMetadata::from_shape(shape).require_non_empty(op)?;
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
    let metadata = AxisReductionMetadata::new(operand.shape(), operand.strides(), axis, keepdims)?;
    metadata.require_non_empty(op)?;
    let mut values = Vec::with_capacity(metadata.output.len);

    for lane_offset in
        offset_iter(operand.offset(), &metadata.output.shape, &metadata.output.outer_strides)
    {
        let mut variance = RunningVariance::default();
        for index in 0..metadata.axis_len {
            variance.add(
                operand.data()[lane_offset + index * metadata.axis_stride]
                    .to_f64()
                    .ok_or(AtlasNdError::NumericConversionFailed { op })?,
            );
        }
        let value = variance.population();
        values.push(if stddev { value.sqrt() } else { value });
    }

    NDArray::from_shape_vec(metadata.output.shape, values)
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
