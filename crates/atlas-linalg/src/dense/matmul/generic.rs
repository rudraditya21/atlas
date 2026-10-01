use atlas_ndarray::Numeric;
use rayon::prelude::*;

use super::dispatch::should_parallelize_matmul;
use crate::internal::dense::{MatrixRef, VectorRef};

pub(super) fn vector_matrix_into<T: Numeric>(
    lhs: VectorRef<'_, T>,
    rhs: MatrixRef<'_, T>,
    output: &mut [T],
) {
    for (col, output) in output.iter_mut().enumerate() {
        let mut total = T::zero();

        for k in 0..lhs.len {
            total += lhs.value_at(k) * rhs.value_at(k, col);
        }

        *output = total;
    }
}

pub(super) fn matrix_vector_into<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: VectorRef<'_, T>,
    output: &mut [T],
) {
    for (row, output) in output.iter_mut().enumerate() {
        let mut total = T::zero();

        for k in 0..lhs.cols {
            total += lhs.value_at(row, k) * rhs.value_at(k);
        }

        *output = total;
    }
}

pub(super) fn matrix_matrix_into<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
    output: &mut [T],
) {
    if should_parallelize_matmul(lhs.rows, lhs.cols, rhs.cols) {
        output.par_chunks_mut(rhs.cols).enumerate().for_each(|(row, out_row)| {
            for (col, output) in out_row.iter_mut().enumerate() {
                let mut total = T::zero();

                for k in 0..lhs.cols {
                    total += lhs.value_at(row, k) * rhs.value_at(k, col);
                }

                *output = total;
            }
        });
    } else {
        for row in 0..lhs.rows {
            let out_row = &mut output[row * rhs.cols..(row + 1) * rhs.cols];

            for (col, output) in out_row.iter_mut().enumerate() {
                let mut total = T::zero();

                for k in 0..lhs.cols {
                    total += lhs.value_at(row, k) * rhs.value_at(k, col);
                }

                *output = total;
            }
        }
    }
}
