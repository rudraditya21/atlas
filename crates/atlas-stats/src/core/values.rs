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
    let len = operand.len()?;

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
    let len = operand.len()?;
    if len == 0 {
        return Ok(());
    }

    operand.try_for_each_span(|values| {
        for value in values {
            let value = value.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op })?;
            f(value)?;
        }

        Ok(())
    })
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
    lhs.try_for_each_span_pair(rhs, |lhs_values, rhs_values| {
        for (left, right) in lhs_values.iter().zip(rhs_values) {
            let left = left.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op })?;
            let right = right.to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op })?;
            f(left, right)?;
        }

        Ok(())
    })
}
