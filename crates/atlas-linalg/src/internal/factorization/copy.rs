use atlas_ndarray::Numeric;

use crate::core::LinalgOperand;

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
