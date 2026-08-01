use atlas_ndarray::{NDArray, Numeric};
use rayon::prelude::*;

use crate::core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand};
use crate::internal::dense::{
    MatrixRef, VectorRef, dot_contiguous, dot_kernel, matrix_ref, vector_ref,
};
use crate::internal::simd;

const ROW_MAJOR_MATMUL_BLOCK_SIZE: usize = 32;
const ROW_MAJOR_MATMUL_BLOCK_THRESHOLD: usize = 64 * 64 * 64;
const PARALLEL_MATMUL_THRESHOLD: usize = 128 * 128 * 128;
const PARALLEL_MATMUL_MIN_ROWS_PER_THREAD: usize = 16;

pub fn matmul<'a, T, L, R>(lhs: L, rhs: R) -> AtlasLinalgResult<NDArray<T>>
where
    T: Numeric + 'a,
    L: Into<LinalgOperand<'a, T>>,
    R: Into<LinalgOperand<'a, T>>,
{
    let lhs = lhs.into();
    let rhs = rhs.into();

    match (lhs.ndim(), rhs.ndim()) {
        (1, 1) => matmul_vector_vector(&lhs, &rhs),
        (1, 2) => matmul_vector_matrix(&lhs, &rhs),
        (2, 1) => matmul_matrix_vector(&lhs, &rhs),
        (2, 2) => matmul_matrix_matrix(&lhs, &rhs),
        _ => Err(AtlasLinalgError::InvalidOperandRank {
            op: "matmul",
            left: lhs.ndim(),
            right: rhs.ndim(),
        }),
    }
}

fn matmul_vector_vector<T: Numeric>(
    lhs: &LinalgOperand<'_, T>,
    rhs: &LinalgOperand<'_, T>,
) -> AtlasLinalgResult<NDArray<T>> {
    let lhs = vector_ref(lhs);
    let rhs = vector_ref(rhs);

    if lhs.len != rhs.len {
        return Err(AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: vec![lhs.len],
            right: vec![rhs.len],
            reason: "vector lengths must match",
        });
    }

    let value = dot_kernel(lhs, rhs);
    wrap_scalar(value)
}

fn matmul_vector_matrix<T: Numeric>(
    lhs: &LinalgOperand<'_, T>,
    rhs: &LinalgOperand<'_, T>,
) -> AtlasLinalgResult<NDArray<T>> {
    let lhs = vector_ref(lhs);
    let rhs = matrix_ref(rhs);

    if lhs.len != rhs.rows {
        return Err(AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: vec![lhs.len],
            right: vec![rhs.rows, rhs.cols],
            reason: "vector length must match matrix row count",
        });
    }

    let data = if lhs.is_contiguous() && rhs.is_row_major_contiguous() {
        matmul_vector_matrix_row_major(lhs, rhs)
    } else if lhs.is_contiguous() && rhs.is_col_major_contiguous() {
        matmul_vector_matrix_col_major(lhs, rhs)
    } else {
        matmul_vector_matrix_generic(lhs, rhs)
    };

    Ok(NDArray::from_shape_vec([rhs.cols], data)?)
}

fn matmul_matrix_vector<T: Numeric>(
    lhs: &LinalgOperand<'_, T>,
    rhs: &LinalgOperand<'_, T>,
) -> AtlasLinalgResult<NDArray<T>> {
    let lhs = matrix_ref(lhs);
    let rhs = vector_ref(rhs);

    if lhs.cols != rhs.len {
        return Err(AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: vec![lhs.rows, lhs.cols],
            right: vec![rhs.len],
            reason: "matrix column count must match vector length",
        });
    }

    let data = if lhs.is_row_major_contiguous() && rhs.is_contiguous() {
        matmul_matrix_vector_row_major(lhs, rhs)
    } else if lhs.is_col_major_contiguous() && rhs.is_contiguous() {
        matmul_matrix_vector_col_major(lhs, rhs)
    } else {
        matmul_matrix_vector_generic(lhs, rhs)
    };

    Ok(NDArray::from_shape_vec([lhs.rows], data)?)
}

fn matmul_matrix_matrix<T: Numeric>(
    lhs: &LinalgOperand<'_, T>,
    rhs: &LinalgOperand<'_, T>,
) -> AtlasLinalgResult<NDArray<T>> {
    let lhs = matrix_ref(lhs);
    let rhs = matrix_ref(rhs);

    if lhs.cols != rhs.rows {
        return Err(AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: vec![lhs.rows, lhs.cols],
            right: vec![rhs.rows, rhs.cols],
            reason: "left matrix column count must match right matrix row count",
        });
    }

    let data = if lhs.is_row_major_contiguous() && rhs.is_row_major_contiguous() {
        matmul_matrix_matrix_row_major(lhs, rhs)
    } else if lhs.is_col_major_contiguous() && rhs.is_row_major_contiguous() {
        matmul_matrix_matrix_lhs_col_major(lhs, rhs)
    } else if lhs.is_row_major_contiguous() && rhs.is_col_major_contiguous() {
        matmul_matrix_matrix_rhs_col_major(lhs, rhs)
    } else {
        matmul_matrix_matrix_generic(lhs, rhs)
    };

    Ok(NDArray::from_shape_vec([lhs.rows, rhs.cols], data)?)
}

fn wrap_scalar<T: Numeric>(value: T) -> AtlasLinalgResult<NDArray<T>> {
    Ok(NDArray::from_shape_vec([], vec![value])?)
}

fn matmul_vector_matrix_row_major<T: Numeric>(
    lhs: VectorRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
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

fn matmul_vector_matrix_col_major<T: Numeric>(
    lhs: VectorRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let lhs = lhs.contiguous_slice();
    let mut data = vec![T::zero(); rhs.cols];

    if simd::is_f32::<T>() {
        let lhs = simd::cast_slice::<T, f32>(lhs);
        {
            let data_f32 = simd::cast_mut_slice::<T, f32>(data.as_mut_slice());

            for (col, output) in data_f32.iter_mut().enumerate() {
                *output = simd::dot_contiguous_f32(
                    lhs,
                    simd::cast_slice::<T, f32>(rhs.contiguous_col_slice(col)),
                );
            }
        }

        return data;
    }

    if simd::is_f64::<T>() {
        let lhs = simd::cast_slice::<T, f64>(lhs);
        {
            let data_f64 = simd::cast_mut_slice::<T, f64>(data.as_mut_slice());

            for (col, output) in data_f64.iter_mut().enumerate() {
                *output = simd::dot_contiguous_f64(
                    lhs,
                    simd::cast_slice::<T, f64>(rhs.contiguous_col_slice(col)),
                );
            }
        }

        return data;
    }

    for (col, output) in data.iter_mut().enumerate() {
        *output = dot_contiguous(lhs, rhs.contiguous_col_slice(col));
    }

    data
}

fn matmul_vector_matrix_generic<T: Numeric>(
    lhs: VectorRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
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

fn matmul_matrix_vector_row_major<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: VectorRef<'_, T>,
) -> Vec<T> {
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

fn matmul_matrix_vector_col_major<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: VectorRef<'_, T>,
) -> Vec<T> {
    let mut data = vec![T::zero(); lhs.rows];

    for k in 0..lhs.cols {
        let rhs_value = rhs.value_at(k);
        let lhs_col = lhs.contiguous_col_slice(k);

        for row in 0..lhs.rows {
            data[row] += lhs_col[row] * rhs_value;
        }
    }

    data
}

fn matmul_matrix_vector_generic<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: VectorRef<'_, T>,
) -> Vec<T> {
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

fn matmul_matrix_matrix_row_major<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    if should_use_blocked_row_major_matmul(lhs, rhs) {
        return matmul_matrix_matrix_row_major_blocked(lhs, rhs);
    }

    matmul_matrix_matrix_row_major_simple(lhs, rhs)
}

fn matmul_matrix_matrix_row_major_simple<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let lhs_values = lhs.row_major_region();
    let rhs_values = rhs.row_major_region();
    let mut data = vec![T::zero(); lhs.rows * rhs.cols];

    if simd::is_f32::<T>() {
        return matmul_matrix_matrix_row_major_simple_f32(lhs, rhs, lhs_values, rhs_values, data);
    }

    if simd::is_f64::<T>() {
        return matmul_matrix_matrix_row_major_simple_f64(lhs, rhs, lhs_values, rhs_values, data);
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

fn matmul_matrix_matrix_row_major_blocked<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let lhs_values = lhs.row_major_region();
    let rhs_values = rhs.row_major_region();
    let mut data = vec![T::zero(); lhs.rows * rhs.cols];
    let block = ROW_MAJOR_MATMUL_BLOCK_SIZE;

    if simd::is_f32::<T>() {
        return matmul_matrix_matrix_row_major_blocked_f32(
            lhs, rhs, lhs_values, rhs_values, data, block,
        );
    }

    if simd::is_f64::<T>() {
        return matmul_matrix_matrix_row_major_blocked_f64(
            lhs, rhs, lhs_values, rhs_values, data, block,
        );
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

fn should_parallelize_matmul(rows: usize, inner: usize, cols: usize) -> bool {
    should_parallelize_matmul_for_threads(rows, inner, cols, rayon::current_num_threads())
}

fn should_parallelize_matmul_for_threads(
    rows: usize,
    inner: usize,
    cols: usize,
    thread_count: usize,
) -> bool {
    if thread_count <= 1 {
        return false;
    }

    let work_items = rows.saturating_mul(inner).saturating_mul(cols);

    work_items >= PARALLEL_MATMUL_THRESHOLD
        && rows >= thread_count.saturating_mul(PARALLEL_MATMUL_MIN_ROWS_PER_THREAD)
}

fn matmul_matrix_matrix_lhs_col_major<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let mut data = vec![T::zero(); lhs.rows * rhs.cols];

    if simd::is_f32::<T>() {
        return matmul_matrix_matrix_lhs_col_major_f32(lhs, rhs, data);
    }

    if simd::is_f64::<T>() {
        return matmul_matrix_matrix_lhs_col_major_f64(lhs, rhs, data);
    }

    if should_parallelize_matmul(lhs.rows, lhs.cols, rhs.cols) {
        data.par_chunks_mut(rhs.cols).enumerate().for_each(|(row, out_row)| {
            for k in 0..lhs.cols {
                let lhs_value = lhs.contiguous_col_slice(k)[row];
                let rhs_row = rhs.contiguous_row_slice(k);
                simd::scaled_accumulate_contiguous(out_row, rhs_row, lhs_value);
            }
        });
    } else {
        for k in 0..lhs.cols {
            let lhs_col = lhs.contiguous_col_slice(k);
            let rhs_row = rhs.contiguous_row_slice(k);

            for row in 0..lhs.rows {
                let lhs_value = lhs_col[row];
                let out_row = &mut data[row * rhs.cols..(row + 1) * rhs.cols];
                simd::scaled_accumulate_contiguous(out_row, rhs_row, lhs_value);
            }
        }
    }

    data
}

fn matmul_matrix_matrix_rhs_col_major<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let mut data = vec![T::zero(); lhs.rows * rhs.cols];

    if simd::is_f32::<T>() {
        return matmul_matrix_matrix_rhs_col_major_f32(lhs, rhs, data);
    }

    if simd::is_f64::<T>() {
        return matmul_matrix_matrix_rhs_col_major_f64(lhs, rhs, data);
    }

    if should_parallelize_matmul(lhs.rows, lhs.cols, rhs.cols) {
        data.par_chunks_mut(rhs.cols).enumerate().for_each(|(row, out_row)| {
            let lhs_row = lhs.contiguous_row_slice(row);

            for (col, output) in out_row.iter_mut().enumerate() {
                *output = dot_contiguous(lhs_row, rhs.contiguous_col_slice(col));
            }
        });
    } else {
        for row in 0..lhs.rows {
            let lhs_row = lhs.contiguous_row_slice(row);
            let out_row = &mut data[row * rhs.cols..(row + 1) * rhs.cols];

            for (col, output) in out_row.iter_mut().enumerate() {
                *output = dot_contiguous(lhs_row, rhs.contiguous_col_slice(col));
            }
        }
    }

    data
}

fn matmul_matrix_matrix_generic<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
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

fn matmul_matrix_matrix_row_major_simple_f32<T: Numeric>(
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

fn matmul_matrix_matrix_row_major_simple_f64<T: Numeric>(
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

fn matmul_matrix_matrix_row_major_blocked_f32<T: Numeric>(
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

fn matmul_matrix_matrix_row_major_blocked_f64<T: Numeric>(
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

fn matmul_matrix_matrix_lhs_col_major_f32<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
    mut data: Vec<T>,
) -> Vec<T> {
    {
        let data_f32 = simd::cast_mut_slice::<T, f32>(data.as_mut_slice());

        if should_parallelize_matmul(lhs.rows, lhs.cols, rhs.cols) {
            data_f32.par_chunks_mut(rhs.cols).enumerate().for_each(|(row, out_row)| {
                for k in 0..lhs.cols {
                    let lhs_value = simd::cast_slice::<T, f32>(lhs.contiguous_col_slice(k))[row];
                    let rhs_row = simd::cast_slice::<T, f32>(rhs.contiguous_row_slice(k));
                    simd::scaled_accumulate_contiguous_f32(out_row, rhs_row, lhs_value);
                }
            });
        } else {
            for k in 0..lhs.cols {
                let lhs_col = simd::cast_slice::<T, f32>(lhs.contiguous_col_slice(k));
                let rhs_row = simd::cast_slice::<T, f32>(rhs.contiguous_row_slice(k));

                for row in 0..lhs.rows {
                    let out_row = &mut data_f32[row * rhs.cols..(row + 1) * rhs.cols];
                    simd::scaled_accumulate_contiguous_f32(out_row, rhs_row, lhs_col[row]);
                }
            }
        }
    }

    data
}

fn matmul_matrix_matrix_lhs_col_major_f64<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
    mut data: Vec<T>,
) -> Vec<T> {
    {
        let data_f64 = simd::cast_mut_slice::<T, f64>(data.as_mut_slice());

        if should_parallelize_matmul(lhs.rows, lhs.cols, rhs.cols) {
            data_f64.par_chunks_mut(rhs.cols).enumerate().for_each(|(row, out_row)| {
                for k in 0..lhs.cols {
                    let lhs_value = simd::cast_slice::<T, f64>(lhs.contiguous_col_slice(k))[row];
                    let rhs_row = simd::cast_slice::<T, f64>(rhs.contiguous_row_slice(k));
                    simd::scaled_accumulate_contiguous_f64(out_row, rhs_row, lhs_value);
                }
            });
        } else {
            for k in 0..lhs.cols {
                let lhs_col = simd::cast_slice::<T, f64>(lhs.contiguous_col_slice(k));
                let rhs_row = simd::cast_slice::<T, f64>(rhs.contiguous_row_slice(k));

                for row in 0..lhs.rows {
                    let out_row = &mut data_f64[row * rhs.cols..(row + 1) * rhs.cols];
                    simd::scaled_accumulate_contiguous_f64(out_row, rhs_row, lhs_col[row]);
                }
            }
        }
    }

    data
}

fn matmul_matrix_matrix_rhs_col_major_f32<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
    mut data: Vec<T>,
) -> Vec<T> {
    {
        let data_f32 = simd::cast_mut_slice::<T, f32>(data.as_mut_slice());

        if should_parallelize_matmul(lhs.rows, lhs.cols, rhs.cols) {
            data_f32.par_chunks_mut(rhs.cols).enumerate().for_each(|(row, out_row)| {
                let lhs_row = simd::cast_slice::<T, f32>(lhs.contiguous_row_slice(row));

                for (col, output) in out_row.iter_mut().enumerate() {
                    *output = simd::dot_contiguous_f32(
                        lhs_row,
                        simd::cast_slice::<T, f32>(rhs.contiguous_col_slice(col)),
                    );
                }
            });
        } else {
            for row in 0..lhs.rows {
                let lhs_row = simd::cast_slice::<T, f32>(lhs.contiguous_row_slice(row));
                let out_row = &mut data_f32[row * rhs.cols..(row + 1) * rhs.cols];

                for (col, output) in out_row.iter_mut().enumerate() {
                    *output = simd::dot_contiguous_f32(
                        lhs_row,
                        simd::cast_slice::<T, f32>(rhs.contiguous_col_slice(col)),
                    );
                }
            }
        }
    }

    data
}

fn matmul_matrix_matrix_rhs_col_major_f64<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
    mut data: Vec<T>,
) -> Vec<T> {
    {
        let data_f64 = simd::cast_mut_slice::<T, f64>(data.as_mut_slice());

        if should_parallelize_matmul(lhs.rows, lhs.cols, rhs.cols) {
            data_f64.par_chunks_mut(rhs.cols).enumerate().for_each(|(row, out_row)| {
                let lhs_row = simd::cast_slice::<T, f64>(lhs.contiguous_row_slice(row));

                for (col, output) in out_row.iter_mut().enumerate() {
                    *output = simd::dot_contiguous_f64(
                        lhs_row,
                        simd::cast_slice::<T, f64>(rhs.contiguous_col_slice(col)),
                    );
                }
            });
        } else {
            for row in 0..lhs.rows {
                let lhs_row = simd::cast_slice::<T, f64>(lhs.contiguous_row_slice(row));
                let out_row = &mut data_f64[row * rhs.cols..(row + 1) * rhs.cols];

                for (col, output) in out_row.iter_mut().enumerate() {
                    *output = simd::dot_contiguous_f64(
                        lhs_row,
                        simd::cast_slice::<T, f64>(rhs.contiguous_col_slice(col)),
                    );
                }
            }
        }
    }

    data
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::{
        matmul, matmul_matrix_matrix_generic, matmul_matrix_matrix_lhs_col_major,
        matmul_matrix_matrix_rhs_col_major, matmul_matrix_matrix_row_major,
        matmul_matrix_vector_col_major, matmul_matrix_vector_generic,
        matmul_matrix_vector_row_major, matmul_vector_matrix_col_major,
        matmul_vector_matrix_generic, matmul_vector_matrix_row_major,
    };
    use crate::internal::dense::{matrix_ref, vector_ref};
    use crate::{AtlasLinalgError, LinalgOperand};

    fn vector_row_major(values: &[i32]) -> NDArray<i32> {
        NDArray::from_shape_vec([values.len()], values.to_vec()).unwrap()
    }

    fn matrix_row_major(rows: usize, cols: usize, values: &[i32]) -> NDArray<i32> {
        NDArray::from_shape_vec([rows, cols], values.to_vec()).unwrap()
    }

    fn matrix_col_major_from_rows(rows: usize, cols: usize, values: &[i32]) -> NDArray<i32> {
        let mut data = Vec::with_capacity(rows * cols);

        for col in 0..cols {
            for row in 0..rows {
                data.push(values[row * cols + col]);
            }
        }

        NDArray::from_shape_vec([cols, rows], data).unwrap()
    }

    fn matrix_generic_from_rows(rows: usize, cols: usize, values: &[i32]) -> NDArray<i32> {
        let padded_cols = cols + 1;
        let mut data = vec![0_i32; rows * padded_cols];

        for row in 0..rows {
            for col in 0..cols {
                data[row * padded_cols + col] = values[row * cols + col];
            }
        }

        NDArray::from_shape_vec([rows, padded_cols], data).unwrap()
    }

    #[test]
    fn matmul_supports_vector_and_matrix_operands() {
        let lhs_vec = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
        let rhs_vec = NDArray::from_shape_vec([3], vec![4_i32, 5, 6]).unwrap();
        let matrix = NDArray::from_shape_vec([3, 2], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let left_matrix = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let right_matrix = NDArray::from_shape_vec([3, 2], vec![7_i32, 8, 9, 10, 11, 12]).unwrap();

        assert_eq!(matmul(&lhs_vec, &rhs_vec).unwrap().shape(), &[] as &[usize]);
        assert_eq!(matmul(&lhs_vec, &rhs_vec).unwrap().data(), &[32]);
        assert_eq!(matmul(&lhs_vec, &matrix).unwrap().data(), &[22, 28]);
        assert_eq!(matmul(&left_matrix, &rhs_vec).unwrap().data(), &[32, 77]);
        assert_eq!(matmul(&left_matrix, &right_matrix).unwrap().shape(), &[2, 2]);
        assert_eq!(matmul(&left_matrix, &right_matrix).unwrap().data(), &[58, 64, 139, 154]);
    }

    #[test]
    fn matmul_vector_vector_reports_matmul_shape_mismatch() {
        let lhs = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
        let rhs = NDArray::from_shape_vec([2], vec![4_i32, 5]).unwrap();

        assert_eq!(
            matmul(&lhs, &rhs).unwrap_err(),
            AtlasLinalgError::ShapeMismatch {
                op: "matmul",
                left: vec![3],
                right: vec![2],
                reason: "vector lengths must match",
            }
        );
    }

    #[test]
    fn matmul_supports_transposed_views_and_strided_fallbacks() {
        let left_base = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let rhs = NDArray::from_shape_vec([2, 2], vec![7_i32, 8, 9, 10]).unwrap();
        let rhs_base = NDArray::from_shape_vec([2, 3], vec![7_i32, 9, 11, 8, 10, 12]).unwrap();
        let generic_base = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let vector = NDArray::from_shape_vec([2], vec![10_i32, 20]).unwrap();

        let left_transposed = left_base.view().transpose();
        let right_transposed = rhs_base.view().transpose();
        let sliced = generic_base.view().slice([0, 1], [2, 2]).unwrap();

        assert_eq!(matmul(left_transposed, &rhs).unwrap().data(), &[43, 48, 59, 66, 75, 84]);
        assert_eq!(matmul(&left_base, right_transposed).unwrap().data(), &[58, 64, 139, 154]);
        assert_eq!(matmul(sliced, &vector).unwrap().data(), &[50, 140]);
    }

    #[test]
    fn vector_matrix_kernels_match_row_major_col_major_and_generic_layouts() {
        let lhs = vector_row_major(&[1, 2, 3]);
        let rhs_values = [7_i32, 8, 9, 10, 11, 12];
        let rhs_row_major = matrix_row_major(3, 2, &rhs_values);
        let rhs_col_major_base = matrix_col_major_from_rows(3, 2, &rhs_values);
        let rhs_generic_base = matrix_generic_from_rows(3, 2, &rhs_values);

        let lhs_operand = LinalgOperand::from(&lhs);
        let rhs_row_major_operand = LinalgOperand::from(&rhs_row_major);
        let rhs_col_major_operand = LinalgOperand::from(rhs_col_major_base.view().transpose());
        let rhs_generic_operand =
            LinalgOperand::from(rhs_generic_base.view().slice([0, 0], [3, 2]).unwrap());

        let row_major = matmul_vector_matrix_row_major(
            vector_ref(&lhs_operand),
            matrix_ref(&rhs_row_major_operand),
        );
        let col_major = matmul_vector_matrix_col_major(
            vector_ref(&lhs_operand),
            matrix_ref(&rhs_col_major_operand),
        );
        let generic = matmul_vector_matrix_generic(
            vector_ref(&lhs_operand),
            matrix_ref(&rhs_generic_operand),
        );

        assert_eq!(row_major, col_major);
        assert_eq!(row_major, generic);
    }

    #[test]
    fn matrix_vector_kernels_match_row_major_col_major_and_generic_layouts() {
        let lhs_values = [1_i32, 2, 3, 4, 5, 6];
        let lhs_row_major = matrix_row_major(2, 3, &lhs_values);
        let lhs_col_major_base = matrix_col_major_from_rows(2, 3, &lhs_values);
        let lhs_generic_base = matrix_generic_from_rows(2, 3, &lhs_values);
        let rhs = vector_row_major(&[7, 8, 9]);

        let lhs_row_major_operand = LinalgOperand::from(&lhs_row_major);
        let lhs_col_major_operand = LinalgOperand::from(lhs_col_major_base.view().transpose());
        let lhs_generic_operand =
            LinalgOperand::from(lhs_generic_base.view().slice([0, 0], [2, 3]).unwrap());
        let rhs_operand = LinalgOperand::from(&rhs);

        let row_major = matmul_matrix_vector_row_major(
            matrix_ref(&lhs_row_major_operand),
            vector_ref(&rhs_operand),
        );
        let col_major = matmul_matrix_vector_col_major(
            matrix_ref(&lhs_col_major_operand),
            vector_ref(&rhs_operand),
        );
        let generic = matmul_matrix_vector_generic(
            matrix_ref(&lhs_generic_operand),
            vector_ref(&rhs_operand),
        );

        assert_eq!(row_major, col_major);
        assert_eq!(row_major, generic);
    }

    #[test]
    fn matrix_matrix_kernels_match_specialized_and_generic_layouts() {
        let lhs_values = [1_i32, 2, 3, 4, 5, 6];
        let rhs_values = [7_i32, 8, 9, 10, 11, 12];
        let lhs_row_major = matrix_row_major(2, 3, &lhs_values);
        let rhs_row_major = matrix_row_major(3, 2, &rhs_values);
        let lhs_col_major_base = matrix_col_major_from_rows(2, 3, &lhs_values);
        let rhs_col_major_base = matrix_col_major_from_rows(3, 2, &rhs_values);
        let lhs_generic_base = matrix_generic_from_rows(2, 3, &lhs_values);
        let rhs_generic_base = matrix_generic_from_rows(3, 2, &rhs_values);

        let lhs_row_major_operand = LinalgOperand::from(&lhs_row_major);
        let rhs_row_major_operand = LinalgOperand::from(&rhs_row_major);
        let lhs_col_major_operand = LinalgOperand::from(lhs_col_major_base.view().transpose());
        let rhs_col_major_operand = LinalgOperand::from(rhs_col_major_base.view().transpose());
        let lhs_generic_operand =
            LinalgOperand::from(lhs_generic_base.view().slice([0, 0], [2, 3]).unwrap());
        let rhs_generic_operand =
            LinalgOperand::from(rhs_generic_base.view().slice([0, 0], [3, 2]).unwrap());

        let row_major = matmul_matrix_matrix_row_major(
            matrix_ref(&lhs_row_major_operand),
            matrix_ref(&rhs_row_major_operand),
        );
        let lhs_col_major = matmul_matrix_matrix_lhs_col_major(
            matrix_ref(&lhs_col_major_operand),
            matrix_ref(&rhs_row_major_operand),
        );
        let rhs_col_major = matmul_matrix_matrix_rhs_col_major(
            matrix_ref(&lhs_row_major_operand),
            matrix_ref(&rhs_col_major_operand),
        );
        let generic = matmul_matrix_matrix_generic(
            matrix_ref(&lhs_generic_operand),
            matrix_ref(&rhs_generic_operand),
        );

        assert_eq!(row_major, lhs_col_major);
        assert_eq!(row_major, rhs_col_major);
        assert_eq!(row_major, generic);
    }

    #[test]
    fn matrix_matrix_row_major_blocked_path_matches_generic_for_larger_inputs() {
        let side = 80;
        let lhs_values: Vec<i32> = (0..side * side).map(|index| (index % 7) as i32 - 3).collect();
        let rhs_values: Vec<i32> = (0..side * side).map(|index| (index % 5) as i32 + 1).collect();

        let lhs_row_major = matrix_row_major(side, side, &lhs_values);
        let rhs_row_major = matrix_row_major(side, side, &rhs_values);
        let lhs_generic_base = matrix_generic_from_rows(side, side, &lhs_values);
        let rhs_generic_base = matrix_generic_from_rows(side, side, &rhs_values);

        let lhs_row_major_operand = LinalgOperand::from(&lhs_row_major);
        let rhs_row_major_operand = LinalgOperand::from(&rhs_row_major);
        let lhs_generic_operand =
            LinalgOperand::from(lhs_generic_base.view().slice([0, 0], [side, side]).unwrap());
        let rhs_generic_operand =
            LinalgOperand::from(rhs_generic_base.view().slice([0, 0], [side, side]).unwrap());

        let row_major = matmul_matrix_matrix_row_major(
            matrix_ref(&lhs_row_major_operand),
            matrix_ref(&rhs_row_major_operand),
        );
        let generic = matmul_matrix_matrix_generic(
            matrix_ref(&lhs_generic_operand),
            matrix_ref(&rhs_generic_operand),
        );

        assert_eq!(row_major, generic);
    }

    #[test]
    fn matmul_dispatch_stays_serial_for_small_and_medium_inputs() {
        assert!(!super::should_parallelize_matmul_for_threads(64, 64, 64, 8));
        assert!(!super::should_parallelize_matmul_for_threads(128, 128, 128, 16));
        assert!(!super::should_parallelize_matmul_for_threads(128, 128, 128, 1));
    }

    #[test]
    fn matmul_dispatch_requires_enough_rows_per_thread() {
        assert!(super::should_parallelize_matmul_for_threads(128, 128, 128, 8));
        assert!(!super::should_parallelize_matmul_for_threads(120, 128, 128, 8));
    }
}
