use std::arch::aarch64::*;

use super::Tile;

pub(super) fn run_f32(mut tile: Tile<'_, f32>) {
    const ROWS: usize = 4;
    const LANES: usize = 4;
    let row_body = tile.rows / ROWS * ROWS;
    let col_body = tile.cols / LANES * LANES;

    for row in (0..row_body).step_by(ROWS) {
        for col in (0..col_body).step_by(LANES) {
            unsafe {
                let mut accumulators = [
                    vld1q_f32(tile.output.as_ptr().add(row * tile.output_stride + col)),
                    vld1q_f32(tile.output.as_ptr().add((row + 1) * tile.output_stride + col)),
                    vld1q_f32(tile.output.as_ptr().add((row + 2) * tile.output_stride + col)),
                    vld1q_f32(tile.output.as_ptr().add((row + 3) * tile.output_stride + col)),
                ];
                for k in 0..tile.inner {
                    let right = vld1q_f32(tile.rhs.as_ptr().add(k * tile.cols + col));
                    for (local_row, accumulator) in accumulators.iter_mut().enumerate() {
                        *accumulator = vfmaq_n_f32(
                            *accumulator,
                            right,
                            tile.lhs[(row + local_row) * tile.inner + k],
                        );
                    }
                }
                for (local_row, accumulator) in accumulators.into_iter().enumerate() {
                    vst1q_f32(
                        tile.output.as_mut_ptr().add((row + local_row) * tile.output_stride + col),
                        accumulator,
                    );
                }
            }
        }
    }

    let rows = tile.rows;
    let cols = tile.cols;
    scalar_tail(&mut tile, 0..row_body, col_body..cols);
    scalar_tail(&mut tile, row_body..rows, 0..cols);
}

pub(super) fn run_f64(mut tile: Tile<'_, f64>) {
    const ROWS: usize = 4;
    const LANES: usize = 2;
    let row_body = tile.rows / ROWS * ROWS;
    let col_body = tile.cols / LANES * LANES;

    for row in (0..row_body).step_by(ROWS) {
        for col in (0..col_body).step_by(LANES) {
            unsafe {
                let mut accumulators = [
                    vld1q_f64(tile.output.as_ptr().add(row * tile.output_stride + col)),
                    vld1q_f64(tile.output.as_ptr().add((row + 1) * tile.output_stride + col)),
                    vld1q_f64(tile.output.as_ptr().add((row + 2) * tile.output_stride + col)),
                    vld1q_f64(tile.output.as_ptr().add((row + 3) * tile.output_stride + col)),
                ];
                for k in 0..tile.inner {
                    let right = vld1q_f64(tile.rhs.as_ptr().add(k * tile.cols + col));
                    for (local_row, accumulator) in accumulators.iter_mut().enumerate() {
                        *accumulator = vfmaq_n_f64(
                            *accumulator,
                            right,
                            tile.lhs[(row + local_row) * tile.inner + k],
                        );
                    }
                }
                for (local_row, accumulator) in accumulators.into_iter().enumerate() {
                    vst1q_f64(
                        tile.output.as_mut_ptr().add((row + local_row) * tile.output_stride + col),
                        accumulator,
                    );
                }
            }
        }
    }

    let rows = tile.rows;
    let cols = tile.cols;
    scalar_tail(&mut tile, 0..row_body, col_body..cols);
    scalar_tail(&mut tile, row_body..rows, 0..cols);
}

fn scalar_tail<T: Copy + std::ops::AddAssign + std::ops::Mul<Output = T>>(
    tile: &mut Tile<'_, T>,
    rows: std::ops::Range<usize>,
    columns: std::ops::Range<usize>,
) {
    for row in rows {
        for col in columns.clone() {
            let mut value = tile.output[row * tile.output_stride + col];
            for k in 0..tile.inner {
                value += tile.lhs[row * tile.inner + k] * tile.rhs[k * tile.cols + col];
            }
            tile.output[row * tile.output_stride + col] = value;
        }
    }
}
