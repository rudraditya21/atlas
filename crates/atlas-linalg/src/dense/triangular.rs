use atlas_ndarray::{NDArray, Numeric};
use num_traits::Float;

use crate::core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand};
use crate::internal::factorization::{tolerance, validate_rank_two};

pub fn solve_lower_triangular<'a, 'b, T, M, R>(matrix: M, rhs: R) -> AtlasLinalgResult<NDArray<T>>
where
    T: Numeric + Float + 'a + 'b,
    M: Into<LinalgOperand<'a, T>>,
    R: Into<LinalgOperand<'b, T>>,
{
    let matrix = matrix.into();
    let (rows, columns) = validate_rank_two(&matrix, "solve_lower_triangular")?;

    if rows != columns {
        return Err(AtlasLinalgError::InvalidInputShape {
            op: "solve_lower_triangular",
            shape: matrix.shape().to_vec(),
            reason: "lower-triangular coefficient matrix must be square",
        });
    }

    let rhs = rhs.into();
    let (rhs_columns, vector_rhs) = match rhs.shape() {
        [rhs_rows] if *rhs_rows == rows => (1, true),
        [rhs_rows, rhs_columns] if *rhs_rows == rows => (*rhs_columns, false),
        [..] if rhs.ndim() == 1 || rhs.ndim() == 2 => {
            return Err(AtlasLinalgError::ShapeMismatch {
                op: "solve_lower_triangular",
                left: vec![rows, columns],
                right: rhs.shape().to_vec(),
                reason: "right-hand side row count must match coefficient matrix row count",
            });
        }
        _ => {
            return Err(AtlasLinalgError::InvalidInputRank {
                op: "solve_lower_triangular",
                expected: "a vector or matrix",
                rank: rhs.ndim(),
            });
        }
    };
    let mut values = vec![T::zero(); rows * rhs_columns];

    for row in 0..rows {
        let diagonal = matrix_value(&matrix, row, row);
        if diagonal.abs() <= tolerance::<T>() {
            return Err(AtlasLinalgError::SingularMatrix {
                op: "solve_lower_triangular",
                pivot: row,
            });
        }

        for rhs_column in 0..rhs_columns {
            let mut value = rhs_value(&rhs, row, rhs_column);
            for previous_row in 0..row {
                value -= matrix_value(&matrix, row, previous_row)
                    * values[previous_row * rhs_columns + rhs_column];
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
