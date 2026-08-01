mod cholesky;
mod lu;
mod qr;

use atlas_ndarray::Numeric;
use num_traits::Float;

use crate::core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand};

pub use cholesky::{CholeskyFactorization, cholesky};
pub use lu::{LuFactorization, lu};
pub use qr::{QrFactorization, qr};

pub(super) fn validate_rank_two<T: Numeric>(
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

pub(super) fn copy_matrix_row_major<T: Numeric>(operand: &LinalgOperand<'_, T>) -> Vec<T> {
    let rows = operand.shape()[0];
    let cols = operand.shape()[1];
    let row_stride = operand.strides()[0];
    let col_stride = operand.strides()[1];
    let offset = operand.offset();
    let data = operand.data();
    let mut copied = Vec::with_capacity(rows * cols);

    for row in 0..rows {
        for col in 0..cols {
            copied.push(data[offset + row * row_stride + col * col_stride]);
        }
    }

    copied
}

pub(super) fn zero_matrix_data<T: Numeric>(rows: usize, cols: usize) -> Vec<T> {
    vec![T::zero(); rows * cols]
}

pub(super) fn identity_matrix_data<T: Numeric>(n: usize) -> Vec<T> {
    let mut data = zero_matrix_data(n, n);

    for index in 0..n {
        data[index * n + index] = T::one();
    }

    data
}

pub(super) fn find_pivot_row<T: Float>(matrix: &[T], n: usize, pivot_col: usize) -> usize {
    let mut pivot_row = pivot_col;
    let mut pivot_value = matrix[pivot_col * n + pivot_col].abs();

    for candidate in (pivot_col + 1)..n {
        let value = matrix[candidate * n + pivot_col].abs();

        if value > pivot_value {
            pivot_value = value;
            pivot_row = candidate;
        }
    }

    pivot_row
}

pub(super) fn swap_rows<T>(matrix: &mut [T], cols: usize, left: usize, right: usize) {
    for col in 0..cols {
        matrix.swap(left * cols + col, right * cols + col);
    }
}

pub(super) fn swap_l_prefix_rows<T>(
    matrix: &mut [T],
    cols: usize,
    left: usize,
    right: usize,
    end: usize,
) {
    for col in 0..end {
        matrix.swap(left * cols + col, right * cols + col);
    }
}

pub(super) fn extract_column<T: Copy>(
    matrix: &[T],
    rows: usize,
    cols: usize,
    column: usize,
) -> Vec<T> {
    let mut values = Vec::with_capacity(rows);

    for row in 0..rows {
        values.push(matrix[row * cols + column]);
    }

    values
}

pub(super) fn column_from_storage<T: Copy>(
    matrix: &[T],
    rows: usize,
    cols: usize,
    column: usize,
) -> Vec<T> {
    let mut values = Vec::with_capacity(rows);

    for row in 0..rows {
        values.push(matrix[row * cols + column]);
    }

    values
}

pub(super) fn dot_slice<T: Numeric>(lhs: &[T], rhs: &[T]) -> T {
    let mut total = T::zero();

    for index in 0..lhs.len() {
        total += lhs[index] * rhs[index];
    }

    total
}

pub(super) fn vector_norm<T: Numeric + Float>(values: &[T]) -> T {
    dot_slice(values, values).sqrt()
}

pub(super) fn is_symmetric<T: Float>(matrix: &[T], n: usize, tolerance: T) -> bool {
    for row in 0..n {
        for col in 0..row {
            let diff = (matrix[row * n + col] - matrix[col * n + row]).abs();

            if diff > tolerance {
                return false;
            }
        }
    }

    true
}

pub(super) fn tolerance<T: Float>() -> T {
    T::epsilon().sqrt()
}
