use std::arch::aarch64::*;

pub(crate) fn dot_f32(lhs: &[f32], rhs: &[f32]) -> f32 {
    let mut accumulator = unsafe { vdupq_n_f32(0.0) };
    let mut index = 0;
    while index + 4 <= lhs.len() {
        unsafe {
            accumulator = vfmaq_f32(
                accumulator,
                vld1q_f32(lhs.as_ptr().add(index)),
                vld1q_f32(rhs.as_ptr().add(index)),
            );
        }
        index += 4;
    }
    let mut total = unsafe { vaddvq_f32(accumulator) };
    for index in index..lhs.len() {
        total += lhs[index] * rhs[index];
    }
    total
}

pub(crate) fn dot_f64(lhs: &[f64], rhs: &[f64]) -> f64 {
    let mut accumulator = unsafe { vdupq_n_f64(0.0) };
    let mut index = 0;
    while index + 2 <= lhs.len() {
        unsafe {
            accumulator = vfmaq_f64(
                accumulator,
                vld1q_f64(lhs.as_ptr().add(index)),
                vld1q_f64(rhs.as_ptr().add(index)),
            );
        }
        index += 2;
    }
    let mut total = unsafe { vaddvq_f64(accumulator) };
    for index in index..lhs.len() {
        total += lhs[index] * rhs[index];
    }
    total
}

pub(crate) fn squared_euclidean_f64(lhs: &[f64], rhs: &[f64]) -> f64 {
    let mut accumulator = unsafe { vdupq_n_f64(0.0) };
    let mut index = 0;
    while index + 2 <= lhs.len() {
        unsafe {
            let delta =
                vsubq_f64(vld1q_f64(lhs.as_ptr().add(index)), vld1q_f64(rhs.as_ptr().add(index)));
            accumulator = vfmaq_f64(accumulator, delta, delta);
        }
        index += 2;
    }
    let mut total = unsafe { vaddvq_f64(accumulator) };
    for index in index..lhs.len() {
        let delta = lhs[index] - rhs[index];
        total += delta * delta;
    }
    total
}

pub(crate) fn scaled_accumulate_f32(output: &mut [f32], input: &[f32], scale: f32) {
    let scale = unsafe { vdupq_n_f32(scale) };
    let mut index = 0;
    while index + 4 <= output.len() {
        unsafe {
            let accumulated = vld1q_f32(output.as_ptr().add(index));
            let values = vld1q_f32(input.as_ptr().add(index));
            vst1q_f32(output.as_mut_ptr().add(index), vfmaq_f32(accumulated, values, scale));
        }
        index += 4;
    }
    for index in index..output.len() {
        output[index] += input[index] * scale;
    }
}

pub(crate) fn scaled_accumulate_f64(output: &mut [f64], input: &[f64], scale: f64) {
    let scale = unsafe { vdupq_n_f64(scale) };
    let mut index = 0;
    while index + 2 <= output.len() {
        unsafe {
            let accumulated = vld1q_f64(output.as_ptr().add(index));
            let values = vld1q_f64(input.as_ptr().add(index));
            vst1q_f64(output.as_mut_ptr().add(index), vfmaq_f64(accumulated, values, scale));
        }
        index += 2;
    }
    for index in index..output.len() {
        output[index] += input[index] * scale;
    }
}

pub(crate) fn matmul_tile_f32(
    lhs: &[f32],
    rhs: &[f32],
    output: &mut [f32],
    output_stride: usize,
    rows: usize,
    inner: usize,
    cols: usize,
) {
    const ROWS: usize = 4;
    const LANES: usize = 4;
    let row_body = rows / ROWS * ROWS;
    let col_body = cols / LANES * LANES;

    for row in (0..row_body).step_by(ROWS) {
        for col in (0..col_body).step_by(LANES) {
            unsafe {
                let mut accumulator0 = vld1q_f32(output.as_ptr().add(row * output_stride + col));
                let mut accumulator1 =
                    vld1q_f32(output.as_ptr().add((row + 1) * output_stride + col));
                let mut accumulator2 =
                    vld1q_f32(output.as_ptr().add((row + 2) * output_stride + col));
                let mut accumulator3 =
                    vld1q_f32(output.as_ptr().add((row + 3) * output_stride + col));

                for k in 0..inner {
                    let right = vld1q_f32(rhs.as_ptr().add(k * cols + col));
                    accumulator0 = vfmaq_n_f32(accumulator0, right, lhs[row * inner + k]);
                    accumulator1 = vfmaq_n_f32(accumulator1, right, lhs[(row + 1) * inner + k]);
                    accumulator2 = vfmaq_n_f32(accumulator2, right, lhs[(row + 2) * inner + k]);
                    accumulator3 = vfmaq_n_f32(accumulator3, right, lhs[(row + 3) * inner + k]);
                }

                vst1q_f32(output.as_mut_ptr().add(row * output_stride + col), accumulator0);
                vst1q_f32(output.as_mut_ptr().add((row + 1) * output_stride + col), accumulator1);
                vst1q_f32(output.as_mut_ptr().add((row + 2) * output_stride + col), accumulator2);
                vst1q_f32(output.as_mut_ptr().add((row + 3) * output_stride + col), accumulator3);
            }
        }
    }

    accumulate_tail_f32(lhs, rhs, output, output_stride, inner, cols, 0..row_body, col_body..cols);
    accumulate_tail_f32(lhs, rhs, output, output_stride, inner, cols, row_body..rows, 0..cols);
}

pub(crate) fn matmul_tile_f64(
    lhs: &[f64],
    rhs: &[f64],
    output: &mut [f64],
    output_stride: usize,
    rows: usize,
    inner: usize,
    cols: usize,
) {
    const ROWS: usize = 4;
    const LANES: usize = 2;
    let row_body = rows / ROWS * ROWS;
    let col_body = cols / LANES * LANES;

    for row in (0..row_body).step_by(ROWS) {
        for col in (0..col_body).step_by(LANES) {
            unsafe {
                let mut accumulator0 = vld1q_f64(output.as_ptr().add(row * output_stride + col));
                let mut accumulator1 =
                    vld1q_f64(output.as_ptr().add((row + 1) * output_stride + col));
                let mut accumulator2 =
                    vld1q_f64(output.as_ptr().add((row + 2) * output_stride + col));
                let mut accumulator3 =
                    vld1q_f64(output.as_ptr().add((row + 3) * output_stride + col));

                for k in 0..inner {
                    let right = vld1q_f64(rhs.as_ptr().add(k * cols + col));
                    accumulator0 = vfmaq_n_f64(accumulator0, right, lhs[row * inner + k]);
                    accumulator1 = vfmaq_n_f64(accumulator1, right, lhs[(row + 1) * inner + k]);
                    accumulator2 = vfmaq_n_f64(accumulator2, right, lhs[(row + 2) * inner + k]);
                    accumulator3 = vfmaq_n_f64(accumulator3, right, lhs[(row + 3) * inner + k]);
                }

                vst1q_f64(output.as_mut_ptr().add(row * output_stride + col), accumulator0);
                vst1q_f64(output.as_mut_ptr().add((row + 1) * output_stride + col), accumulator1);
                vst1q_f64(output.as_mut_ptr().add((row + 2) * output_stride + col), accumulator2);
                vst1q_f64(output.as_mut_ptr().add((row + 3) * output_stride + col), accumulator3);
            }
        }
    }

    accumulate_tail_f64(lhs, rhs, output, output_stride, inner, cols, 0..row_body, col_body..cols);
    accumulate_tail_f64(lhs, rhs, output, output_stride, inner, cols, row_body..rows, 0..cols);
}

fn accumulate_tail_f32(
    lhs: &[f32],
    rhs: &[f32],
    output: &mut [f32],
    output_stride: usize,
    inner: usize,
    cols: usize,
    rows: std::ops::Range<usize>,
    columns: std::ops::Range<usize>,
) {
    for row in rows {
        for col in columns.clone() {
            let mut value = output[row * output_stride + col];
            for k in 0..inner {
                value += lhs[row * inner + k] * rhs[k * cols + col];
            }
            output[row * output_stride + col] = value;
        }
    }
}

fn accumulate_tail_f64(
    lhs: &[f64],
    rhs: &[f64],
    output: &mut [f64],
    output_stride: usize,
    inner: usize,
    cols: usize,
    rows: std::ops::Range<usize>,
    columns: std::ops::Range<usize>,
) {
    for row in rows {
        for col in columns.clone() {
            let mut value = output[row * output_stride + col];
            for k in 0..inner {
                value += lhs[row * inner + k] * rhs[k * cols + col];
            }
            output[row * output_stride + col] = value;
        }
    }
}
