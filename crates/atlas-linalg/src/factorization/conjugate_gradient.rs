use atlas_ndarray::{NDArray, Numeric};
use num_traits::Float;

use crate::{
    core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand},
    internal::factorization::{
        copy_matrix_row_major, dot_slice, is_symmetric, tolerance as matrix_tolerance,
        validate_finite, validate_rank_two,
    },
};

const OP: &str = "conjugate_gradient";

/// Solves a symmetric positive-definite linear system with conjugate gradients.
///
/// `tolerance` is the absolute Euclidean residual-norm threshold.
pub fn conjugate_gradient<'a, 'b, T, M, R>(
    matrix: M,
    rhs: R,
    max_iterations: usize,
    tolerance: T,
) -> AtlasLinalgResult<NDArray<T>>
where
    T: Numeric + Float + 'a + 'b,
    M: Into<LinalgOperand<'a, T>>,
    R: Into<LinalgOperand<'b, T>>,
{
    if max_iterations == 0 {
        return Err(AtlasLinalgError::InvalidArgument {
            op: OP,
            reason: "maximum iterations must be positive",
        });
    }
    if !tolerance.is_finite() || tolerance < T::zero() {
        return Err(AtlasLinalgError::InvalidArgument {
            op: OP,
            reason: "tolerance must be finite and nonnegative",
        });
    }

    let matrix = matrix.into();
    let (rows, columns) = validate_rank_two(&matrix, OP)?;
    if rows != columns {
        return Err(AtlasLinalgError::InvalidInputShape {
            op: OP,
            shape: matrix.shape().to_vec(),
            reason: "coefficient matrix must be square",
        });
    }

    let rhs = rhs.into();
    if rhs.ndim() != 1 {
        return Err(AtlasLinalgError::InvalidInputRank {
            op: OP,
            expected: "a rank-1 right-hand-side vector",
            rank: rhs.ndim(),
        });
    }
    if rhs.shape()[0] != rows {
        return Err(AtlasLinalgError::ShapeMismatch {
            op: OP,
            left: vec![rows, columns],
            right: rhs.shape().to_vec(),
            reason: "right-hand side length must match coefficient matrix order",
        });
    }

    let coefficients = copy_matrix_row_major(&matrix);
    validate_finite(&coefficients, OP)?;
    if !is_symmetric(&coefficients, rows, matrix_tolerance::<T>()) {
        return Err(AtlasLinalgError::InvalidInputShape {
            op: OP,
            shape: matrix.shape().to_vec(),
            reason: "coefficient matrix must be symmetric",
        });
    }
    let mut residual = logical_vector(&rhs);
    validate_finite(&residual, OP)?;

    let mut solution = vec![T::zero(); rows];
    let mut direction = residual.clone();
    let mut residual_norm_squared = dot_slice(&residual, &residual);
    if residual_norm_squared.sqrt() <= tolerance {
        return NDArray::from_shape_vec([rows], solution).map_err(Into::into);
    }

    for iteration in 0..max_iterations {
        let matrix_direction = matrix_vector_product(&coefficients, rows, &direction);
        let direction_product = dot_slice(&direction, &matrix_direction);
        if direction_product <= T::zero() {
            return Err(AtlasLinalgError::NotPositiveDefinite { op: OP, index: iteration });
        }

        let step = residual_norm_squared / direction_product;
        for index in 0..rows {
            solution[index] = solution[index] + step * direction[index];
            residual[index] = residual[index] - step * matrix_direction[index];
        }

        let next_residual_norm_squared = dot_slice(&residual, &residual);
        if next_residual_norm_squared.sqrt() <= tolerance {
            return NDArray::from_shape_vec([rows], solution).map_err(Into::into);
        }
        let beta = next_residual_norm_squared / residual_norm_squared;
        for index in 0..rows {
            direction[index] = residual[index] + beta * direction[index];
        }
        residual_norm_squared = next_residual_norm_squared;
    }

    Err(AtlasLinalgError::IterationLimit { op: OP, iterations: max_iterations })
}

fn logical_vector<T: Numeric>(vector: &LinalgOperand<'_, T>) -> Vec<T> {
    (0..vector.shape()[0])
        .map(|index| vector.data()[vector.offset() + index * vector.strides()[0]])
        .collect()
}

fn matrix_vector_product<T: Numeric>(matrix: &[T], rows: usize, vector: &[T]) -> Vec<T> {
    (0..rows).map(|row| dot_slice(&matrix[row * rows..(row + 1) * rows], vector)).collect()
}
