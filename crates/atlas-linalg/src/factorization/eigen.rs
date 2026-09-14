use atlas_ndarray::{NDArray, Numeric};
use num_traits::Float;

use crate::{
    core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand},
    internal::factorization::{
        copy_matrix_row_major, is_symmetric, tolerance, validate_finite, validate_rank_two,
    },
};

const OP: &str = "symmetric_eigendecomposition";

/// Eigenvalues and orthonormal eigenvectors of a real symmetric matrix.
///
/// Eigenvalues are ascending; each corresponding eigenvector is stored as a column.
pub struct SymmetricEigenDecomposition<T: Numeric> {
    eigenvalues: NDArray<T>,
    eigenvectors: NDArray<T>,
}

impl<T: Numeric> SymmetricEigenDecomposition<T> {
    pub fn eigenvalues(&self) -> &NDArray<T> {
        &self.eigenvalues
    }

    pub fn eigenvectors(&self) -> &NDArray<T> {
        &self.eigenvectors
    }
}

/// Computes the eigendecomposition of a finite real symmetric matrix with Jacobi rotations.
pub fn symmetric_eigendecomposition<'a, T, M>(
    matrix: M,
) -> AtlasLinalgResult<SymmetricEigenDecomposition<T>>
where
    T: Numeric + Float + 'a,
    M: Into<LinalgOperand<'a, T>>,
{
    let matrix = matrix.into();
    let (rows, columns) = validate_rank_two(&matrix, OP)?;
    if rows != columns {
        return Err(AtlasLinalgError::InvalidInputShape {
            op: OP,
            shape: matrix.shape().to_vec(),
            reason: "coefficient matrix must be square",
        });
    }

    let order = rows;
    let mut values = copy_matrix_row_major(&matrix);
    validate_finite(&values, OP)?;
    let convergence_tolerance = tolerance::<T>();
    if !is_symmetric(&values, order, convergence_tolerance) {
        return Err(AtlasLinalgError::InvalidInputShape {
            op: OP,
            shape: matrix.shape().to_vec(),
            reason: "coefficient matrix must be symmetric",
        });
    }

    let mut eigenvectors = identity(order);
    let max_iterations = order.saturating_mul(order).saturating_mul(100);
    for iteration in 0..max_iterations {
        let Some((row, column, largest)) = largest_off_diagonal(&values, order) else {
            break;
        };
        if largest <= convergence_tolerance {
            return decomposition(values, eigenvectors, order);
        }

        jacobi_rotate(&mut values, &mut eigenvectors, order, row, column);
        if iteration + 1 == max_iterations {
            return Err(AtlasLinalgError::IterationLimit { op: OP, iterations: max_iterations });
        }
    }

    decomposition(values, eigenvectors, order)
}

fn decomposition<T: Numeric + Float>(
    values: Vec<T>,
    eigenvectors: Vec<T>,
    order: usize,
) -> AtlasLinalgResult<SymmetricEigenDecomposition<T>> {
    validate_finite(&values, OP)?;
    let (eigenvalues, eigenvectors) = sort_eigenpairs(values, eigenvectors, order);
    Ok(SymmetricEigenDecomposition {
        eigenvalues: NDArray::from_shape_vec([order], eigenvalues)?,
        eigenvectors: NDArray::from_shape_vec([order, order], eigenvectors)?,
    })
}

fn identity<T: Numeric>(order: usize) -> Vec<T> {
    let mut matrix = vec![T::zero(); order * order];
    for index in 0..order {
        matrix[index * order + index] = T::one();
    }
    matrix
}

fn largest_off_diagonal<T: Float>(matrix: &[T], order: usize) -> Option<(usize, usize, T)> {
    let mut largest = None;
    for row in 0..order {
        for column in (row + 1)..order {
            let value = matrix[row * order + column].abs();
            if largest.as_ref().map_or(true, |(_, _, current)| value > *current) {
                largest = Some((row, column, value));
            }
        }
    }
    largest
}

fn jacobi_rotate<T: Numeric + Float>(
    matrix: &mut [T],
    eigenvectors: &mut [T],
    order: usize,
    row: usize,
    column: usize,
) {
    let diagonal_row = matrix[row * order + row];
    let diagonal_column = matrix[column * order + column];
    let off_diagonal = matrix[row * order + column];
    let tau = (diagonal_column - diagonal_row) / (T::one() + T::one()) / off_diagonal;
    let rotation = if tau >= T::zero() {
        T::one() / (tau + (T::one() + tau * tau).sqrt())
    } else {
        -T::one() / (-tau + (T::one() + tau * tau).sqrt())
    };
    let cosine = T::one() / (T::one() + rotation * rotation).sqrt();
    let sine = rotation * cosine;

    for index in 0..order {
        if index == row || index == column {
            continue;
        }
        let row_value = matrix[index * order + row];
        let column_value = matrix[index * order + column];
        let rotated_row = cosine * row_value - sine * column_value;
        let rotated_column = sine * row_value + cosine * column_value;
        matrix[index * order + row] = rotated_row;
        matrix[row * order + index] = rotated_row;
        matrix[index * order + column] = rotated_column;
        matrix[column * order + index] = rotated_column;
    }

    matrix[row * order + row] = cosine * cosine * diagonal_row
        - (T::one() + T::one()) * sine * cosine * off_diagonal
        + sine * sine * diagonal_column;
    matrix[column * order + column] = sine * sine * diagonal_row
        + (T::one() + T::one()) * sine * cosine * off_diagonal
        + cosine * cosine * diagonal_column;
    matrix[row * order + column] = T::zero();
    matrix[column * order + row] = T::zero();

    for index in 0..order {
        let row_value = eigenvectors[index * order + row];
        let column_value = eigenvectors[index * order + column];
        eigenvectors[index * order + row] = cosine * row_value - sine * column_value;
        eigenvectors[index * order + column] = sine * row_value + cosine * column_value;
    }
}

fn sort_eigenpairs<T: Numeric + Float>(
    matrix: Vec<T>,
    eigenvectors: Vec<T>,
    order: usize,
) -> (Vec<T>, Vec<T>) {
    let mut indices = (0..order).collect::<Vec<_>>();
    indices.sort_unstable_by(|left, right| {
        matrix[left * order + left]
            .partial_cmp(&matrix[right * order + right])
            .expect("eigenvalues remain finite after finite-input Jacobi rotations")
            .then_with(|| left.cmp(right))
    });

    let eigenvalues = indices.iter().map(|&index| matrix[index * order + index]).collect();
    let mut sorted_eigenvectors = vec![T::zero(); order * order];
    for (new_column, old_column) in indices.into_iter().enumerate() {
        for row in 0..order {
            sorted_eigenvectors[row * order + new_column] = eigenvectors[row * order + old_column];
        }
    }

    (eigenvalues, sorted_eigenvectors)
}
