use atlas_ndarray::Numeric;
use num_traits::ToPrimitive;

use crate::core::{AtlasStatsError, AtlasStatsResult, StatsOperand};

pub(crate) fn validate_vector_pair<T: Numeric>(
    lhs: &StatsOperand<'_, T>,
    rhs: &StatsOperand<'_, T>,
    op: &'static str,
) -> AtlasStatsResult<()> {
    if lhs.ndim() != 1 {
        return Err(AtlasStatsError::InvalidInputRank {
            op,
            expected: "rank-1 vector",
            rank: lhs.ndim(),
        });
    }

    if rhs.ndim() != 1 {
        return Err(AtlasStatsError::InvalidInputRank {
            op,
            expected: "rank-1 vector",
            rank: rhs.ndim(),
        });
    }

    if lhs.shape()[0] != rhs.shape()[0] {
        return Err(AtlasStatsError::ShapeMismatch {
            op,
            left: lhs.shape().to_vec(),
            right: rhs.shape().to_vec(),
            reason: "vector lengths must match",
        });
    }

    Ok(())
}

pub(crate) fn validate_non_empty<T>(
    operand: &StatsOperand<'_, T>,
    op: &'static str,
) -> AtlasStatsResult<usize>
where
    T: Numeric,
{
    let len = operand.len();

    if len == 0 {
        return Err(AtlasStatsError::EmptyInput { op });
    }

    Ok(len)
}

pub(crate) fn mean<T>(operand: &StatsOperand<'_, T>, op: &'static str) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive,
{
    let len = validate_non_empty(operand, op)?;
    let mut total = 0.0_f64;

    try_for_each_f64(operand, op, |value| {
        total += value;
        Ok(())
    })?;

    Ok(total / len as f64)
}

pub(crate) fn means<T>(
    lhs: &StatsOperand<'_, T>,
    rhs: &StatsOperand<'_, T>,
    op: &'static str,
) -> AtlasStatsResult<(f64, f64, usize)>
where
    T: Numeric + ToPrimitive,
{
    let len = validate_non_empty(lhs, op)?;
    let mut lhs_total = 0.0_f64;
    let mut rhs_total = 0.0_f64;

    try_for_each_vector_pair_f64(lhs, rhs, op, |left, right| {
        lhs_total += left;
        rhs_total += right;
        Ok(())
    })?;

    Ok((lhs_total / len as f64, rhs_total / len as f64, len))
}

pub(crate) fn try_for_each_f64<T, F>(
    operand: &StatsOperand<'_, T>,
    op: &'static str,
    mut f: F,
) -> AtlasStatsResult<()>
where
    T: Numeric + ToPrimitive,
    F: FnMut(f64) -> AtlasStatsResult<()>,
{
    let len = operand.len();
    if len == 0 {
        return Ok(());
    }

    let shape = operand.shape();
    let strides = operand.strides();
    let data = operand.data();
    let base_offset = operand.offset();

    if shape.is_empty() {
        let value =
            data[base_offset].to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op })?;
        f(value)?;
        return Ok(());
    }

    if shape.len() == 1 {
        let mut offset = base_offset;
        let stride = strides[0];

        for _ in 0..shape[0] {
            let value =
                data[offset].to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op })?;
            f(value)?;
            offset += stride;
        }

        return Ok(());
    }

    if is_contiguous(shape, strides) {
        for index in 0..len {
            let value = data[base_offset + index]
                .to_f64()
                .ok_or(AtlasStatsError::NumericConversionFailed { op })?;
            f(value)?;
        }

        return Ok(());
    }

    let mut index = vec![0usize; shape.len()];

    loop {
        let mut offset = base_offset;

        for axis in 0..shape.len() {
            offset += index[axis] * strides[axis];
        }

        let value = data[offset].to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op })?;
        f(value)?;

        if !advance_index(&mut index, shape) {
            break;
        }
    }

    Ok(())
}

pub(crate) fn try_for_each_vector_pair_f64<T, F>(
    lhs: &StatsOperand<'_, T>,
    rhs: &StatsOperand<'_, T>,
    op: &'static str,
    mut f: F,
) -> AtlasStatsResult<()>
where
    T: Numeric + ToPrimitive,
    F: FnMut(f64, f64) -> AtlasStatsResult<()>,
{
    let len = lhs.shape()[0];
    let lhs_data = lhs.data();
    let rhs_data = rhs.data();
    let mut lhs_offset = lhs.offset();
    let mut rhs_offset = rhs.offset();
    let lhs_stride = lhs.strides()[0];
    let rhs_stride = rhs.strides()[0];

    for _ in 0..len {
        let left =
            lhs_data[lhs_offset].to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op })?;
        let right =
            rhs_data[rhs_offset].to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op })?;
        f(left, right)?;
        lhs_offset += lhs_stride;
        rhs_offset += rhs_stride;
    }

    Ok(())
}

fn advance_index(index: &mut [usize], shape: &[usize]) -> bool {
    for axis in (0..shape.len()).rev() {
        index[axis] += 1;

        if index[axis] < shape[axis] {
            return true;
        }

        index[axis] = 0;
    }

    false
}

fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.is_empty() {
        return true;
    }

    let mut expected = 1;

    for axis in (0..shape.len()).rev() {
        if strides[axis] != expected {
            return false;
        }

        expected *= shape[axis];
    }

    true
}
