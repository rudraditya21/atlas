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
