use atlas_ndarray::{NDArray, Numeric};
use num_traits::Float;

use crate::{
    core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand},
    internal::factorization::{tolerance, validate_rank_two},
};

pub fn solve_lower_triangular<'a, 'b, T, M, R>(matrix: M, rhs: R) -> AtlasLinalgResult<NDArray<T>>
where
    T: Numeric + Float + 'a + 'b,
    M: Into<LinalgOperand<'a, T>>,
    R: Into<LinalgOperand<'b, T>>,
{
    solve_triangular(matrix, rhs, Triangle::Lower, "solve_lower_triangular")
}

pub fn solve_upper_triangular<'a, 'b, T, M, R>(matrix: M, rhs: R) -> AtlasLinalgResult<NDArray<T>>
where
    T: Numeric + Float + 'a + 'b,
    M: Into<LinalgOperand<'a, T>>,
    R: Into<LinalgOperand<'b, T>>,
{
    solve_triangular(matrix, rhs, Triangle::Upper, "solve_upper_triangular")
}

pub(crate) fn solve_lower_triangular_with_op<'a, 'b, T, M, R>(
    matrix: M,
    rhs: R,
    op: &'static str,
) -> AtlasLinalgResult<NDArray<T>>
where
    T: Numeric + Float + 'a + 'b,
    M: Into<LinalgOperand<'a, T>>,
    R: Into<LinalgOperand<'b, T>>,
{
    solve_triangular(matrix, rhs, Triangle::Lower, op)
}

pub(crate) fn solve_upper_triangular_with_op<'a, 'b, T, M, R>(
    matrix: M,
    rhs: R,
    op: &'static str,
) -> AtlasLinalgResult<NDArray<T>>
where
    T: Numeric + Float + 'a + 'b,
    M: Into<LinalgOperand<'a, T>>,
    R: Into<LinalgOperand<'b, T>>,
{
    solve_triangular(matrix, rhs, Triangle::Upper, op)
}

#[derive(Clone, Copy)]
enum Triangle {
    Lower,
    Upper,
}

fn solve_triangular<'a, 'b, T, M, R>(
    matrix: M,
    rhs: R,
    triangle: Triangle,
    op: &'static str,
) -> AtlasLinalgResult<NDArray<T>>
where
    T: Numeric + Float + 'a + 'b,
    M: Into<LinalgOperand<'a, T>>,
    R: Into<LinalgOperand<'b, T>>,
{
    let matrix = matrix.into();
    let (rows, columns) = validate_rank_two(&matrix, op)?;

    if rows != columns {
        return Err(AtlasLinalgError::InvalidInputShape {
            op,
            shape: matrix.shape().to_vec(),
            reason: match triangle {
                Triangle::Lower => "lower-triangular coefficient matrix must be square",
                Triangle::Upper => "upper-triangular coefficient matrix must be square",
            },
        });
    }

    let rhs = rhs.into();
    let (rhs_columns, vector_rhs) = match rhs.shape() {
        [rhs_rows] if *rhs_rows == rows => (1, true),
        [rhs_rows, rhs_columns] if *rhs_rows == rows => (*rhs_columns, false),
        [..] if rhs.ndim() == 1 || rhs.ndim() == 2 => {
            return Err(AtlasLinalgError::ShapeMismatch {
                op,
                left: vec![rows, columns],
                right: rhs.shape().to_vec(),
                reason: "right-hand side row count must match coefficient matrix row count",
            });
        }
        _ => {
            return Err(AtlasLinalgError::InvalidInputRank {
                op,
                expected: "a vector or matrix",
                rank: rhs.ndim(),
            });
        }
    };
    let mut values = vec![T::zero(); rows * rhs_columns];

    for index in 0..rows {
        let row = match triangle {
            Triangle::Lower => index,
            Triangle::Upper => rows - index - 1,
        };
        let diagonal = matrix_value(&matrix, row, row);
        if diagonal.abs() <= tolerance::<T>() {
            return Err(AtlasLinalgError::SingularMatrix { op, pivot: row });
        }

        for rhs_column in 0..rhs_columns {
            let mut value = rhs_value(&rhs, row, rhs_column);
            match triangle {
                Triangle::Lower => {
                    for previous_row in 0..row {
                        value -= matrix_value(&matrix, row, previous_row)
                            * values[previous_row * rhs_columns + rhs_column];
                    }
                }
                Triangle::Upper => {
                    for next_row in (row + 1)..rows {
                        value -= matrix_value(&matrix, row, next_row)
                            * values[next_row * rhs_columns + rhs_column];
                    }
                }
            }
            values[row * rhs_columns + rhs_column] = value / diagonal;
        }
    }

    if vector_rhs {
        NDArray::from_shape_vec([rows], values).map_err(Into::into)
    } else {
        NDArray::from_shape_vec([rows, rhs_columns], values).map_err(Into::into)
    }
}

fn matrix_value<T: Numeric>(matrix: &LinalgOperand<'_, T>, row: usize, column: usize) -> T {
    matrix.data()[matrix.offset() + row * matrix.strides()[0] + column * matrix.strides()[1]]
}

fn rhs_value<T: Numeric>(rhs: &LinalgOperand<'_, T>, row: usize, column: usize) -> T {
    let offset = rhs.offset() + row * rhs.strides()[0];

    if rhs.ndim() == 1 {
        rhs.data()[offset]
    } else {
        rhs.data()[offset + column * rhs.strides()[1]]
    }
}
