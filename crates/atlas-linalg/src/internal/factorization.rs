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

pub(crate) fn copy_matrix_row_major<T: Numeric>(operand: &LinalgOperand<'_, T>) -> Vec<T> {
    let rows = operand.shape()[0];
    let cols = operand.shape()[1];
    let row_stride = operand.strides()[0];
    let col_stride = operand.strides()[1];
    let offset = operand.offset();
    let data = operand.data();

    if col_stride == 1 && row_stride == cols {
        return data[offset..offset + rows * cols].to_vec();
    }

    let mut copied = vec![T::zero(); rows * cols];

    for row in 0..rows {
        let dst_row = &mut copied[row * cols..(row + 1) * cols];
        let src_row_offset = offset + row * row_stride;

        for col in 0..cols {
            dst_row[col] = data[src_row_offset + col * col_stride];
        }
    }

    copied
}

pub(crate) fn zero_matrix_data<T: Numeric>(rows: usize, cols: usize) -> Vec<T> {
    vec![T::zero(); rows * cols]
}

pub(crate) fn identity_matrix_data<T: Numeric>(n: usize) -> Vec<T> {
    let mut data = zero_matrix_data(n, n);

    for index in 0..n {
        data[index * n + index] = T::one();
    }

    data
}

pub(crate) fn find_pivot_row<T: Float>(matrix: &[T], n: usize, pivot_col: usize) -> usize {
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

pub(crate) fn swap_rows<T>(matrix: &mut [T], cols: usize, left: usize, right: usize) {
    if left == right {
        return;
    }

    let left_start = left * cols;
    let right_start = right * cols;
    let (head, tail) = matrix.split_at_mut(right_start);
    let left_row = &mut head[left_start..left_start + cols];
    let right_row = &mut tail[..cols];

    left_row.swap_with_slice(right_row);
}

pub(crate) fn swap_l_prefix_rows<T>(
    matrix: &mut [T],
    cols: usize,
    left: usize,
    right: usize,
    end: usize,
) {
    if left == right || end == 0 {
        return;
    }

    let left_start = left * cols;
    let right_start = right * cols;
    let (head, tail) = matrix.split_at_mut(right_start);
    let left_prefix = &mut head[left_start..left_start + end];
    let right_prefix = &mut tail[..end];

    left_prefix.swap_with_slice(right_prefix);
}

pub(crate) fn dot_slice<T: Numeric>(lhs: &[T], rhs: &[T]) -> T {
    let len = lhs.len();
    let mut acc0 = T::zero();
    let mut acc1 = T::zero();
    let mut acc2 = T::zero();
    let mut acc3 = T::zero();
    let mut index = 0;

    while index + 4 <= len {
        acc0 += lhs[index] * rhs[index];
        acc1 += lhs[index + 1] * rhs[index + 1];
        acc2 += lhs[index + 2] * rhs[index + 2];
        acc3 += lhs[index + 3] * rhs[index + 3];
        index += 4;
    }

    let mut total = acc0 + acc1;
    total += acc2 + acc3;

    while index < len {
        total += lhs[index] * rhs[index];
        index += 1;
    }

    total
}

pub(crate) fn vector_norm<T: Numeric + Float>(values: &[T]) -> T {
    dot_slice(values, values).sqrt()
}

pub(crate) fn is_symmetric<T: Float>(matrix: &[T], n: usize, tolerance: T) -> bool {
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

pub(crate) fn tolerance<T: Float>() -> T {
    T::epsilon().sqrt()
}
