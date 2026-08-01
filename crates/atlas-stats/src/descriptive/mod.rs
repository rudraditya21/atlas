mod correlation;
mod covariance;
mod stddev;
mod variance;

use atlas_ndarray::Numeric;
use num_traits::ToPrimitive;

use crate::core::{AtlasStatsError, AtlasStatsResult, StatsOperand};

pub use correlation::correlation;
pub use covariance::covariance;
pub use stddev::stddev;
pub use variance::variance;

fn validate_vector_pair<T: Numeric>(
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

fn collect_values<T>(operand: &StatsOperand<'_, T>, op: &'static str) -> AtlasStatsResult<Vec<f64>>
where
    T: Numeric + ToPrimitive,
{
    let len = operand.len();

    if len == 0 {
        return Err(AtlasStatsError::EmptyInput { op });
    }

    let shape = operand.shape();
    let strides = operand.strides();
    let data = operand.data();
    let base_offset = operand.offset();
    let mut values = Vec::with_capacity(len);

    if shape.is_empty() {
        values.push(
            data[base_offset].to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op })?,
        );

        return Ok(values);
    }

    let mut index = vec![0usize; shape.len()];

    loop {
        let mut offset = base_offset;

        for axis in 0..shape.len() {
            offset += index[axis] * strides[axis];
        }

        values.push(data[offset].to_f64().ok_or(AtlasStatsError::NumericConversionFailed { op })?);

        if !advance_index(&mut index, shape) {
            break;
        }
    }

    Ok(values)
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

fn mean_of(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}
