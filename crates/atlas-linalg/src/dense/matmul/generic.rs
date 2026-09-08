use atlas_ndarray::Numeric;
use rayon::prelude::*;

use super::dispatch::should_parallelize_matmul;
use crate::internal::dense::{MatrixRef, VectorRef};

pub(super) fn vector_matrix<T: Numeric>(lhs: VectorRef<'_, T>, rhs: MatrixRef<'_, T>) -> Vec<T> {
    let mut data = vec![T::zero(); rhs.cols];

    for (col, output) in data.iter_mut().enumerate() {
        let mut total = T::zero();

        for k in 0..lhs.len {
            total += lhs.value_at(k) * rhs.value_at(k, col);
        }

        *output = total;
    }

    data
}

pub(super) fn matrix_vector<T: Numeric>(lhs: MatrixRef<'_, T>, rhs: VectorRef<'_, T>) -> Vec<T> {
    let mut data = vec![T::zero(); lhs.rows];

    for (row, output) in data.iter_mut().enumerate() {
        let mut total = T::zero();

        for k in 0..lhs.cols {
            total += lhs.value_at(row, k) * rhs.value_at(k);
        }

        *output = total;
    }

    data
}

pub(super) fn matrix_matrix<T: Numeric>(lhs: MatrixRef<'_, T>, rhs: MatrixRef<'_, T>) -> Vec<T> {
    let mut data = vec![T::zero(); lhs.rows * rhs.cols];

    if should_parallelize_matmul(lhs.rows, lhs.cols, rhs.cols) {
        data.par_chunks_mut(rhs.cols).enumerate().for_each(|(row, out_row)| {
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
            let out_row = &mut data[row * rhs.cols..(row + 1) * rhs.cols];

            for (col, output) in out_row.iter_mut().enumerate() {
                let mut total = T::zero();

                for k in 0..lhs.cols {
                    total += lhs.value_at(row, k) * rhs.value_at(k, col);
                }

                *output = total;
            }
        }
    }

    data
}
