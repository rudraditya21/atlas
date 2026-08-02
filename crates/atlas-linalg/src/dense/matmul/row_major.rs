use atlas_ndarray::Numeric;
use rayon::prelude::*;

use crate::internal::dense::{MatrixRef, VectorRef, dot_contiguous};
use crate::internal::simd;

use super::dispatch::should_parallelize_matmul;

const ROW_MAJOR_MATMUL_BLOCK_SIZE: usize = 32;
const ROW_MAJOR_MATMUL_BLOCK_THRESHOLD: usize = 64 * 64 * 64;

pub(super) fn vector_matrix<T: Numeric>(lhs: VectorRef<'_, T>, rhs: MatrixRef<'_, T>) -> Vec<T> {
    let lhs_values = lhs.contiguous_slice();
    let rhs_values = rhs.row_major_region();
    let mut data = vec![T::zero(); rhs.cols];

    if simd::is_f32::<T>() {
        let lhs_values = simd::cast_slice::<T, f32>(lhs_values);
        let rhs_values = simd::cast_slice::<T, f32>(rhs_values);
        {
            let data_f32 = simd::cast_mut_slice::<T, f32>(data.as_mut_slice());

            for (&lhs_value, rhs_row) in lhs_values.iter().zip(rhs_values.chunks_exact(rhs.cols)) {
                simd::scaled_accumulate_contiguous_f32(data_f32, rhs_row, lhs_value);
            }
        }

        return data;
    }

    if simd::is_f64::<T>() {
        let lhs_values = simd::cast_slice::<T, f64>(lhs_values);
        let rhs_values = simd::cast_slice::<T, f64>(rhs_values);
        {
            let data_f64 = simd::cast_mut_slice::<T, f64>(data.as_mut_slice());

            for (&lhs_value, rhs_row) in lhs_values.iter().zip(rhs_values.chunks_exact(rhs.cols)) {
                simd::scaled_accumulate_contiguous_f64(data_f64, rhs_row, lhs_value);
            }
        }

        return data;
    }

    for (lhs_value, rhs_row) in lhs_values.iter().copied().zip(rhs_values.chunks_exact(rhs.cols)) {
        simd::scaled_accumulate_contiguous(&mut data, rhs_row, lhs_value);
    }

    data
}

pub(super) fn matrix_vector<T: Numeric>(lhs: MatrixRef<'_, T>, rhs: VectorRef<'_, T>) -> Vec<T> {
    let lhs_values = lhs.row_major_region();
    let rhs = rhs.contiguous_slice();
    let mut data = vec![T::zero(); lhs.rows];

    if simd::is_f32::<T>() {
        let lhs_values = simd::cast_slice::<T, f32>(lhs_values);
        let rhs = simd::cast_slice::<T, f32>(rhs);
        {
            let data_f32 = simd::cast_mut_slice::<T, f32>(data.as_mut_slice());

            for (output, lhs_row) in data_f32.iter_mut().zip(lhs_values.chunks_exact(lhs.cols)) {
                *output = simd::dot_contiguous_f32(lhs_row, rhs);
            }
        }

        return data;
    }

    if simd::is_f64::<T>() {
        let lhs_values = simd::cast_slice::<T, f64>(lhs_values);
        let rhs = simd::cast_slice::<T, f64>(rhs);
        {
            let data_f64 = simd::cast_mut_slice::<T, f64>(data.as_mut_slice());

            for (output, lhs_row) in data_f64.iter_mut().zip(lhs_values.chunks_exact(lhs.cols)) {
                *output = simd::dot_contiguous_f64(lhs_row, rhs);
            }
        }

        return data;
    }

    for (output, lhs_row) in data.iter_mut().zip(lhs_values.chunks_exact(lhs.cols)) {
        *output = dot_contiguous(lhs_row, rhs);
    }

    data
}

pub(super) fn matrix_matrix<T: Numeric>(lhs: MatrixRef<'_, T>, rhs: MatrixRef<'_, T>) -> Vec<T> {
    if should_use_blocked_row_major_matmul(lhs, rhs) {
        return matrix_matrix_blocked(lhs, rhs);
    }

    matrix_matrix_simple(lhs, rhs)
}

fn matrix_matrix_simple<T: Numeric>(lhs: MatrixRef<'_, T>, rhs: MatrixRef<'_, T>) -> Vec<T> {
    let lhs_values = lhs.row_major_region();
    let rhs_values = rhs.row_major_region();
    let mut data = vec![T::zero(); lhs.rows * rhs.cols];

    if simd::is_f32::<T>() {
        return matrix_matrix_simple_f32(lhs, rhs, lhs_values, rhs_values, data);
    }

    if simd::is_f64::<T>() {
        return matrix_matrix_simple_f64(lhs, rhs, lhs_values, rhs_values, data);
    }

    if should_parallelize_matmul(lhs.rows, lhs.cols, rhs.cols) {
        data.par_chunks_mut(rhs.cols).enumerate().for_each(|(row, out_row)| {
            let lhs_row = &lhs_values[row * lhs.cols..(row + 1) * lhs.cols];

            for (k, lhs_value) in lhs_row.iter().copied().enumerate() {
                let rhs_row = &rhs_values[k * rhs.cols..(k + 1) * rhs.cols];
                simd::scaled_accumulate_contiguous(out_row, rhs_row, lhs_value);
            }
        });
    } else {
        for (lhs_row, out_row) in
            lhs_values.chunks_exact(lhs.cols).zip(data.chunks_exact_mut(rhs.cols))
        {
            for (k, lhs_value) in lhs_row.iter().copied().enumerate() {
                let rhs_row = &rhs_values[k * rhs.cols..(k + 1) * rhs.cols];
                simd::scaled_accumulate_contiguous(out_row, rhs_row, lhs_value);
            }
        }
    }

    data
}

fn matrix_matrix_blocked<T: Numeric>(lhs: MatrixRef<'_, T>, rhs: MatrixRef<'_, T>) -> Vec<T> {
    let lhs_values = lhs.row_major_region();
    let rhs_values = rhs.row_major_region();
    let mut data = vec![T::zero(); lhs.rows * rhs.cols];
    let block = ROW_MAJOR_MATMUL_BLOCK_SIZE;

    if simd::is_f32::<T>() {
        return matrix_matrix_blocked_f32(lhs, rhs, lhs_values, rhs_values, data, block);
    }

    if simd::is_f64::<T>() {
        return matrix_matrix_blocked_f64(lhs, rhs, lhs_values, rhs_values, data, block);
    }

    if should_parallelize_matmul(lhs.rows, lhs.cols, rhs.cols) {
        data.par_chunks_mut(rhs.cols * block).enumerate().for_each(|(block_index, out_block)| {
            let row_block = block_index * block;
            let row_count = out_block.len() / rhs.cols;
            let row_end = row_block + row_count;

            for k_block in (0..lhs.cols).step_by(block) {
                let k_end = (k_block + block).min(lhs.cols);

                for col_block in (0..rhs.cols).step_by(block) {
                    let col_end = (col_block + block).min(rhs.cols);

                    for row in row_block..row_end {
                        let lhs_row = &lhs_values[row * lhs.cols..(row + 1) * lhs.cols];
                        let local_row = row - row_block;
                        let out_row = &mut out_block
                            [local_row * rhs.cols + col_block..local_row * rhs.cols + col_end];

                        for (local_k, lhs_value) in
                            lhs_row[k_block..k_end].iter().copied().enumerate()
                        {
                            let k = k_block + local_k;
                            let rhs_row =
                                &rhs_values[k * rhs.cols + col_block..k * rhs.cols + col_end];
                            simd::scaled_accumulate_contiguous(out_row, rhs_row, lhs_value);
                        }
                    }
                }
            }
        });
    } else {
        for row_block in (0..lhs.rows).step_by(block) {
            let row_end = (row_block + block).min(lhs.rows);

            for k_block in (0..lhs.cols).step_by(block) {
                let k_end = (k_block + block).min(lhs.cols);

                for col_block in (0..rhs.cols).step_by(block) {
                    let col_end = (col_block + block).min(rhs.cols);

                    for row in row_block..row_end {
                        let lhs_row = &lhs_values[row * lhs.cols..(row + 1) * lhs.cols];
                        let out_row =
                            &mut data[row * rhs.cols + col_block..row * rhs.cols + col_end];

                        for (local_k, lhs_value) in
                            lhs_row[k_block..k_end].iter().copied().enumerate()
                        {
                            let k = k_block + local_k;
                            let rhs_row =
                                &rhs_values[k * rhs.cols + col_block..k * rhs.cols + col_end];
                            simd::scaled_accumulate_contiguous(out_row, rhs_row, lhs_value);
                        }
                    }
                }
            }
        }
    }

    data
}

fn should_use_blocked_row_major_matmul<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> bool {
    lhs.rows * lhs.cols * rhs.cols >= ROW_MAJOR_MATMUL_BLOCK_THRESHOLD
}

fn matrix_matrix_simple_f32<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
    lhs_values: &[T],
    rhs_values: &[T],
    mut data: Vec<T>,
) -> Vec<T> {
    let lhs_values = simd::cast_slice::<T, f32>(lhs_values);
    let rhs_values = simd::cast_slice::<T, f32>(rhs_values);
    {
        let data_f32 = simd::cast_mut_slice::<T, f32>(data.as_mut_slice());

        if should_parallelize_matmul(lhs.rows, lhs.cols, rhs.cols) {
            data_f32.par_chunks_mut(rhs.cols).enumerate().for_each(|(row, out_row)| {
                let lhs_row = &lhs_values[row * lhs.cols..(row + 1) * lhs.cols];

                for (k, lhs_value) in lhs_row.iter().copied().enumerate() {
                    let rhs_row = &rhs_values[k * rhs.cols..(k + 1) * rhs.cols];
                    simd::scaled_accumulate_contiguous_f32(out_row, rhs_row, lhs_value);
                }
            });
        } else {
            for (lhs_row, out_row) in
                lhs_values.chunks_exact(lhs.cols).zip(data_f32.chunks_exact_mut(rhs.cols))
            {
                for (k, lhs_value) in lhs_row.iter().copied().enumerate() {
                    let rhs_row = &rhs_values[k * rhs.cols..(k + 1) * rhs.cols];
                    simd::scaled_accumulate_contiguous_f32(out_row, rhs_row, lhs_value);
                }
            }
        }
    }

    data
}

fn matrix_matrix_simple_f64<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
    lhs_values: &[T],
    rhs_values: &[T],
    mut data: Vec<T>,
) -> Vec<T> {
    let lhs_values = simd::cast_slice::<T, f64>(lhs_values);
    let rhs_values = simd::cast_slice::<T, f64>(rhs_values);
    {
        let data_f64 = simd::cast_mut_slice::<T, f64>(data.as_mut_slice());

        if should_parallelize_matmul(lhs.rows, lhs.cols, rhs.cols) {
            data_f64.par_chunks_mut(rhs.cols).enumerate().for_each(|(row, out_row)| {
                let lhs_row = &lhs_values[row * lhs.cols..(row + 1) * lhs.cols];

                for (k, lhs_value) in lhs_row.iter().copied().enumerate() {
                    let rhs_row = &rhs_values[k * rhs.cols..(k + 1) * rhs.cols];
                    simd::scaled_accumulate_contiguous_f64(out_row, rhs_row, lhs_value);
                }
            });
        } else {
            for (lhs_row, out_row) in
                lhs_values.chunks_exact(lhs.cols).zip(data_f64.chunks_exact_mut(rhs.cols))
            {
                for (k, lhs_value) in lhs_row.iter().copied().enumerate() {
                    let rhs_row = &rhs_values[k * rhs.cols..(k + 1) * rhs.cols];
                    simd::scaled_accumulate_contiguous_f64(out_row, rhs_row, lhs_value);
                }
            }
        }
    }

    data
}

fn matrix_matrix_blocked_f32<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
    lhs_values: &[T],
    rhs_values: &[T],
    mut data: Vec<T>,
    block: usize,
) -> Vec<T> {
    let lhs_values = simd::cast_slice::<T, f32>(lhs_values);
    let rhs_values = simd::cast_slice::<T, f32>(rhs_values);
    {
        let data_f32 = simd::cast_mut_slice::<T, f32>(data.as_mut_slice());

        if should_parallelize_matmul(lhs.rows, lhs.cols, rhs.cols) {
            data_f32.par_chunks_mut(rhs.cols * block).enumerate().for_each(
                |(block_index, out_block)| {
                    let row_block = block_index * block;
                    let row_count = out_block.len() / rhs.cols;
                    let row_end = row_block + row_count;

                    for k_block in (0..lhs.cols).step_by(block) {
                        let k_end = (k_block + block).min(lhs.cols);

                        for col_block in (0..rhs.cols).step_by(block) {
                            let col_end = (col_block + block).min(rhs.cols);

                            for row in row_block..row_end {
                                let lhs_row = &lhs_values[row * lhs.cols..(row + 1) * lhs.cols];
                                let local_row = row - row_block;
                                let out_row = &mut out_block[local_row * rhs.cols + col_block
                                    ..local_row * rhs.cols + col_end];

                                for (local_k, lhs_value) in
                                    lhs_row[k_block..k_end].iter().copied().enumerate()
                                {
                                    let k = k_block + local_k;
                                    let rhs_row = &rhs_values
                                        [k * rhs.cols + col_block..k * rhs.cols + col_end];
                                    simd::scaled_accumulate_contiguous_f32(
                                        out_row, rhs_row, lhs_value,
                                    );
                                }
                            }
                        }
                    }
                },
            );
        } else {
            for row_block in (0..lhs.rows).step_by(block) {
                let row_end = (row_block + block).min(lhs.rows);

                for k_block in (0..lhs.cols).step_by(block) {
                    let k_end = (k_block + block).min(lhs.cols);

                    for col_block in (0..rhs.cols).step_by(block) {
                        let col_end = (col_block + block).min(rhs.cols);

                        for row in row_block..row_end {
                            let lhs_row = &lhs_values[row * lhs.cols..(row + 1) * lhs.cols];
                            let out_row =
                                &mut data_f32[row * rhs.cols + col_block..row * rhs.cols + col_end];

                            for (local_k, lhs_value) in
                                lhs_row[k_block..k_end].iter().copied().enumerate()
                            {
                                let k = k_block + local_k;
                                let rhs_row =
                                    &rhs_values[k * rhs.cols + col_block..k * rhs.cols + col_end];
                                simd::scaled_accumulate_contiguous_f32(out_row, rhs_row, lhs_value);
                            }
                        }
                    }
                }
            }
        }
    }

    data
}

fn matrix_matrix_blocked_f64<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
    lhs_values: &[T],
    rhs_values: &[T],
    mut data: Vec<T>,
    block: usize,
) -> Vec<T> {
    let lhs_values = simd::cast_slice::<T, f64>(lhs_values);
    let rhs_values = simd::cast_slice::<T, f64>(rhs_values);
    {
        let data_f64 = simd::cast_mut_slice::<T, f64>(data.as_mut_slice());

        if should_parallelize_matmul(lhs.rows, lhs.cols, rhs.cols) {
            data_f64.par_chunks_mut(rhs.cols * block).enumerate().for_each(
                |(block_index, out_block)| {
                    let row_block = block_index * block;
                    let row_count = out_block.len() / rhs.cols;
                    let row_end = row_block + row_count;

                    for k_block in (0..lhs.cols).step_by(block) {
                        let k_end = (k_block + block).min(lhs.cols);

                        for col_block in (0..rhs.cols).step_by(block) {
                            let col_end = (col_block + block).min(rhs.cols);

                            for row in row_block..row_end {
                                let lhs_row = &lhs_values[row * lhs.cols..(row + 1) * lhs.cols];
                                let local_row = row - row_block;
                                let out_row = &mut out_block[local_row * rhs.cols + col_block
                                    ..local_row * rhs.cols + col_end];

                                for (local_k, lhs_value) in
                                    lhs_row[k_block..k_end].iter().copied().enumerate()
                                {
                                    let k = k_block + local_k;
                                    let rhs_row = &rhs_values
                                        [k * rhs.cols + col_block..k * rhs.cols + col_end];
                                    simd::scaled_accumulate_contiguous_f64(
                                        out_row, rhs_row, lhs_value,
                                    );
                                }
                            }
                        }
                    }
                },
            );
        } else {
            for row_block in (0..lhs.rows).step_by(block) {
                let row_end = (row_block + block).min(lhs.rows);

                for k_block in (0..lhs.cols).step_by(block) {
                    let k_end = (k_block + block).min(lhs.cols);

                    for col_block in (0..rhs.cols).step_by(block) {
                        let col_end = (col_block + block).min(rhs.cols);

                        for row in row_block..row_end {
                            let lhs_row = &lhs_values[row * lhs.cols..(row + 1) * lhs.cols];
                            let out_row =
                                &mut data_f64[row * rhs.cols + col_block..row * rhs.cols + col_end];

                            for (local_k, lhs_value) in
                                lhs_row[k_block..k_end].iter().copied().enumerate()
                            {
                                let k = k_block + local_k;
                                let rhs_row =
                                    &rhs_values[k * rhs.cols + col_block..k * rhs.cols + col_end];
                                simd::scaled_accumulate_contiguous_f64(out_row, rhs_row, lhs_value);
                            }
                        }
                    }
                }
            }
        }
    }

    data
}
