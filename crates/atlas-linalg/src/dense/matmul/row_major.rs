use std::cell::RefCell;

use atlas_ndarray::Numeric;
use rayon::prelude::*;

use super::dispatch::should_parallelize_matmul;
use crate::internal::{
    dense::{MatrixRef, VectorRef, dot_contiguous},
    simd,
};

const ROW_MAJOR_MATMUL_BLOCK_SIZE: usize = 32;
// The blocked traversal starts once workloads move beyond the medium square
// regime covered by the baseline Criterion benches. This keeps 64x64 work on
// the lighter simple kernel while moving 96x96 and larger products onto the
// cache-friendlier blocked path.
const ROW_MAJOR_MATMUL_BLOCK_THRESHOLD: usize = 96 * 96 * 96;

#[derive(Default)]
struct RowMajorMatmulScratch<T> {
    lhs_block: Vec<T>,
    rhs_panel: Vec<T>,
}

impl<T> RowMajorMatmulScratch<T> {
    fn with_capacity(block: usize) -> Self {
        let capacity = block * block;

        Self { lhs_block: Vec::with_capacity(capacity), rhs_panel: Vec::with_capacity(capacity) }
    }

    fn ensure_capacity(&mut self, block: usize) {
        let capacity = block * block;
        reserve_scratch_buffer(&mut self.lhs_block, capacity);
        reserve_scratch_buffer(&mut self.rhs_panel, capacity);
    }
}

fn reserve_scratch_buffer<T>(buffer: &mut Vec<T>, capacity: usize) {
    if buffer.capacity() < capacity {
        buffer.reserve(capacity - buffer.capacity());
    }
}

thread_local! {
    static ROW_MAJOR_MATMUL_SCRATCH_F32: RefCell<RowMajorMatmulScratch<f32>> =
        RefCell::new(RowMajorMatmulScratch::default());
    static ROW_MAJOR_MATMUL_SCRATCH_F64: RefCell<RowMajorMatmulScratch<f64>> =
        RefCell::new(RowMajorMatmulScratch::default());
}

fn with_row_major_matmul_scratch_f32<R>(
    block: usize,
    f: impl FnOnce(&mut RowMajorMatmulScratch<f32>) -> R,
) -> R {
    ROW_MAJOR_MATMUL_SCRATCH_F32.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.ensure_capacity(block);
        f(&mut scratch)
    })
}

fn with_row_major_matmul_scratch_f64<R>(
    block: usize,
    f: impl FnOnce(&mut RowMajorMatmulScratch<f64>) -> R,
) -> R {
    ROW_MAJOR_MATMUL_SCRATCH_F64.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.ensure_capacity(block);
        f(&mut scratch)
    })
}

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
            let mut scratch = RowMajorMatmulScratch::with_capacity(block);

            for k_block in (0..lhs.cols).step_by(block) {
                let k_end = (k_block + block).min(lhs.cols);
                let k_width = k_end - k_block;
                pack_lhs_block(
                    lhs_values,
                    lhs.cols,
                    row_block,
                    row_end,
                    k_block,
                    k_end,
                    &mut scratch.lhs_block,
                );

                for col_block in (0..rhs.cols).step_by(block) {
                    let col_end = (col_block + block).min(rhs.cols);
                    let panel_width = col_end - col_block;
                    pack_rhs_panel(
                        rhs_values,
                        rhs.cols,
                        k_block,
                        k_end,
                        col_block,
                        col_end,
                        &mut scratch.rhs_panel,
                    );

                    for local_row in 0..row_count {
                        let lhs_row =
                            &scratch.lhs_block[local_row * k_width..(local_row + 1) * k_width];
                        let out_row = &mut out_block
                            [local_row * rhs.cols + col_block..local_row * rhs.cols + col_end];

                        for (local_k, lhs_value) in lhs_row.iter().copied().enumerate() {
                            let panel_offset = local_k * panel_width;
                            let rhs_row =
                                &scratch.rhs_panel[panel_offset..panel_offset + panel_width];
                            simd::scaled_accumulate_contiguous(out_row, rhs_row, lhs_value);
                        }
                    }
                }
            }
        });
    } else {
        let mut scratch = RowMajorMatmulScratch::with_capacity(block);

        for row_block in (0..lhs.rows).step_by(block) {
            let row_end = (row_block + block).min(lhs.rows);
            let row_count = row_end - row_block;

            for k_block in (0..lhs.cols).step_by(block) {
                let k_end = (k_block + block).min(lhs.cols);
                let k_width = k_end - k_block;
                pack_lhs_block(
                    lhs_values,
                    lhs.cols,
                    row_block,
                    row_end,
                    k_block,
                    k_end,
                    &mut scratch.lhs_block,
                );

                for col_block in (0..rhs.cols).step_by(block) {
                    let col_end = (col_block + block).min(rhs.cols);
                    let panel_width = col_end - col_block;
                    pack_rhs_panel(
                        rhs_values,
                        rhs.cols,
                        k_block,
                        k_end,
                        col_block,
                        col_end,
                        &mut scratch.rhs_panel,
                    );

                    for local_row in 0..row_count {
                        let lhs_row =
                            &scratch.lhs_block[local_row * k_width..(local_row + 1) * k_width];
                        let row = row_block + local_row;
                        let out_row =
                            &mut data[row * rhs.cols + col_block..row * rhs.cols + col_end];

                        for (local_k, lhs_value) in lhs_row.iter().copied().enumerate() {
                            let panel_offset = local_k * panel_width;
                            let rhs_row =
                                &scratch.rhs_panel[panel_offset..panel_offset + panel_width];
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
    should_use_blocked_row_major_matmul_for_dims(lhs.rows, lhs.cols, rhs.cols)
}

pub(super) fn should_use_blocked_row_major_matmul_for_dims(
    rows: usize,
    inner: usize,
    cols: usize,
) -> bool {
    rows.saturating_mul(inner).saturating_mul(cols) >= ROW_MAJOR_MATMUL_BLOCK_THRESHOLD
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
                    with_row_major_matmul_scratch_f32(block, |scratch| {
                        let row_block = block_index * block;
                        let row_count = out_block.len() / rhs.cols;
                        let row_end = row_block + row_count;

                        for k_block in (0..lhs.cols).step_by(block) {
                            let k_end = (k_block + block).min(lhs.cols);
                            let k_width = k_end - k_block;
                            pack_lhs_block(
                                lhs_values,
                                lhs.cols,
                                row_block,
                                row_end,
                                k_block,
                                k_end,
                                &mut scratch.lhs_block,
                            );

                            for col_block in (0..rhs.cols).step_by(block) {
                                let col_end = (col_block + block).min(rhs.cols);
                                let panel_width = col_end - col_block;
                                pack_rhs_panel(
                                    rhs_values,
                                    rhs.cols,
                                    k_block,
                                    k_end,
                                    col_block,
                                    col_end,
                                    &mut scratch.rhs_panel,
                                );

                                for local_row in 0..row_count {
                                    let lhs_row = &scratch.lhs_block
                                        [local_row * k_width..(local_row + 1) * k_width];
                                    let out_row = &mut out_block[local_row * rhs.cols + col_block
                                        ..local_row * rhs.cols + col_end];

                                    for (local_k, lhs_value) in lhs_row.iter().copied().enumerate()
                                    {
                                        let panel_offset = local_k * panel_width;
                                        let rhs_row = &scratch.rhs_panel
                                            [panel_offset..panel_offset + panel_width];
                                        simd::scaled_accumulate_contiguous_f32(
                                            out_row, rhs_row, lhs_value,
                                        );
                                    }
                                }
                            }
                        }
                    });
                },
            );
        } else {
            with_row_major_matmul_scratch_f32(block, |scratch| {
                for row_block in (0..lhs.rows).step_by(block) {
                    let row_end = (row_block + block).min(lhs.rows);
                    let row_count = row_end - row_block;

                    for k_block in (0..lhs.cols).step_by(block) {
                        let k_end = (k_block + block).min(lhs.cols);
                        let k_width = k_end - k_block;
                        pack_lhs_block(
                            lhs_values,
                            lhs.cols,
                            row_block,
                            row_end,
                            k_block,
                            k_end,
                            &mut scratch.lhs_block,
                        );

                        for col_block in (0..rhs.cols).step_by(block) {
                            let col_end = (col_block + block).min(rhs.cols);
                            let panel_width = col_end - col_block;
                            pack_rhs_panel(
                                rhs_values,
                                rhs.cols,
                                k_block,
                                k_end,
                                col_block,
                                col_end,
                                &mut scratch.rhs_panel,
                            );

                            for local_row in 0..row_count {
                                let lhs_row = &scratch.lhs_block
                                    [local_row * k_width..(local_row + 1) * k_width];
                                let row = row_block + local_row;
                                let out_row = &mut data_f32
                                    [row * rhs.cols + col_block..row * rhs.cols + col_end];

                                for (local_k, lhs_value) in lhs_row.iter().copied().enumerate() {
                                    let panel_offset = local_k * panel_width;
                                    let rhs_row = &scratch.rhs_panel
                                        [panel_offset..panel_offset + panel_width];
                                    simd::scaled_accumulate_contiguous_f32(
                                        out_row, rhs_row, lhs_value,
                                    );
                                }
                            }
                        }
                    }
                }
            });
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
                    with_row_major_matmul_scratch_f64(block, |scratch| {
                        let row_block = block_index * block;
                        let row_count = out_block.len() / rhs.cols;
                        let row_end = row_block + row_count;

                        for k_block in (0..lhs.cols).step_by(block) {
                            let k_end = (k_block + block).min(lhs.cols);
                            let k_width = k_end - k_block;
                            pack_lhs_block(
                                lhs_values,
                                lhs.cols,
                                row_block,
                                row_end,
                                k_block,
                                k_end,
                                &mut scratch.lhs_block,
                            );

                            for col_block in (0..rhs.cols).step_by(block) {
                                let col_end = (col_block + block).min(rhs.cols);
                                let panel_width = col_end - col_block;
                                pack_rhs_panel(
                                    rhs_values,
                                    rhs.cols,
                                    k_block,
                                    k_end,
                                    col_block,
                                    col_end,
                                    &mut scratch.rhs_panel,
                                );

                                for local_row in 0..row_count {
                                    let lhs_row = &scratch.lhs_block
                                        [local_row * k_width..(local_row + 1) * k_width];
                                    let out_row = &mut out_block[local_row * rhs.cols + col_block
                                        ..local_row * rhs.cols + col_end];

                                    for (local_k, lhs_value) in lhs_row.iter().copied().enumerate()
                                    {
                                        let panel_offset = local_k * panel_width;
                                        let rhs_row = &scratch.rhs_panel
                                            [panel_offset..panel_offset + panel_width];
                                        simd::scaled_accumulate_contiguous_f64(
                                            out_row, rhs_row, lhs_value,
                                        );
                                    }
                                }
                            }
                        }
                    });
                },
            );
        } else {
            with_row_major_matmul_scratch_f64(block, |scratch| {
                for row_block in (0..lhs.rows).step_by(block) {
                    let row_end = (row_block + block).min(lhs.rows);
                    let row_count = row_end - row_block;

                    for k_block in (0..lhs.cols).step_by(block) {
                        let k_end = (k_block + block).min(lhs.cols);
                        let k_width = k_end - k_block;
                        pack_lhs_block(
                            lhs_values,
                            lhs.cols,
                            row_block,
                            row_end,
                            k_block,
                            k_end,
                            &mut scratch.lhs_block,
                        );

                        for col_block in (0..rhs.cols).step_by(block) {
                            let col_end = (col_block + block).min(rhs.cols);
                            let panel_width = col_end - col_block;
                            pack_rhs_panel(
                                rhs_values,
                                rhs.cols,
                                k_block,
                                k_end,
                                col_block,
                                col_end,
                                &mut scratch.rhs_panel,
                            );

                            for local_row in 0..row_count {
                                let lhs_row = &scratch.lhs_block
                                    [local_row * k_width..(local_row + 1) * k_width];
                                let row = row_block + local_row;
                                let out_row = &mut data_f64
                                    [row * rhs.cols + col_block..row * rhs.cols + col_end];

                                for (local_k, lhs_value) in lhs_row.iter().copied().enumerate() {
                                    let panel_offset = local_k * panel_width;
                                    let rhs_row = &scratch.rhs_panel
                                        [panel_offset..panel_offset + panel_width];
                                    simd::scaled_accumulate_contiguous_f64(
                                        out_row, rhs_row, lhs_value,
                                    );
                                }
                            }
                        }
                    }
                }
            });
        }
    }

    data
}

fn pack_lhs_block<T: Copy>(
    lhs_values: &[T],
    lhs_cols: usize,
    row_block: usize,
    row_end: usize,
    k_block: usize,
    k_end: usize,
    block: &mut Vec<T>,
) {
    block.clear();

    for row in row_block..row_end {
        let row_start = row * lhs_cols + k_block;
        block.extend_from_slice(&lhs_values[row_start..row_start + (k_end - k_block)]);
    }
}

fn pack_rhs_panel<T: Copy>(
    rhs_values: &[T],
    rhs_cols: usize,
    k_block: usize,
    k_end: usize,
    col_block: usize,
    col_end: usize,
    panel: &mut Vec<T>,
) {
    panel.clear();

    for k in k_block..k_end {
        let row_start = k * rhs_cols + col_block;
        panel.extend_from_slice(&rhs_values[row_start..row_start + (col_end - col_block)]);
    }
}
