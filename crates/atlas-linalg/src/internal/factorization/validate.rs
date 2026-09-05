use atlas_ndarray::Numeric;
use num_traits::Float;

use crate::core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand};

pub(crate) fn validate_rank_two<T: Numeric>(
    operand: &LinalgOperand<'_, T>,
    op: &'static str,
) -> AtlasLinalgResult<(usize, usize)> {
    if operand.ndim() != 2 {
        return Err(AtlasLinalgError::InvalidInputRank {
            op,
            expected: "rank-2 matrix",
            rank: operand.ndim(),
        });
    }

    Ok((operand.shape()[0], operand.shape()[1]))
}

pub(crate) fn validate_finite<T: Float>(values: &[T], op: &'static str) -> AtlasLinalgResult<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(AtlasLinalgError::NonFiniteInput { op })
    }
}
