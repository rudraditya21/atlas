use atlas_ndarray::{NDArray, Numeric};
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

pub(crate) fn validate_lower_triangular<T: Numeric>(
    matrix: &NDArray<T>,
    op: &'static str,
    factor: &'static str,
    unit_diagonal: bool,
) -> AtlasLinalgResult<()> {
    validate_triangular(matrix, op, factor, unit_diagonal, Triangle::Lower)
}

pub(crate) fn validate_upper_triangular<T: Numeric>(
    matrix: &NDArray<T>,
    op: &'static str,
    factor: &'static str,
) -> AtlasLinalgResult<()> {
    validate_triangular(matrix, op, factor, false, Triangle::Upper)
}

#[derive(Clone, Copy)]
enum Triangle {
    Lower,
    Upper,
}

fn validate_triangular<T: Numeric>(
    matrix: &NDArray<T>,
    op: &'static str,
    factor: &'static str,
    unit_diagonal: bool,
    triangle: Triangle,
) -> AtlasLinalgResult<()> {
    let [rows, columns] = matrix.shape() else {
        return Err(AtlasLinalgError::InvalidFactor {
            op,
            factor,
            reason: "must be a square matrix",
        });
    };
    if rows != columns {
        return Err(AtlasLinalgError::InvalidFactor {
            op,
            factor,
            reason: "must be a square matrix",
        });
    }

    let order = *rows;
    for row in 0..order {
        if unit_diagonal && matrix.data()[row * order + row] != T::one() {
            return Err(AtlasLinalgError::InvalidFactor {
                op,
                factor,
                reason: "must have a unit diagonal",
            });
        }
        let mut invalid_columns = match triangle {
            Triangle::Lower => (row + 1)..order,
            Triangle::Upper => 0..row,
        };
        if invalid_columns.any(|column| !matrix.data()[row * order + column].is_zero()) {
            return Err(AtlasLinalgError::InvalidFactor {
                op,
                factor,
                reason: "must be triangular",
            });
        }
    }

    Ok(())
}
