use std::arch::aarch64::*;

macro_rules! binary_kernel {
    ($name:ident, $ty:ty, $lanes:expr, $load:ident, $store:ident, $op:ident, $tail:tt) => {
        pub(crate) fn $name(lhs: &[$ty], rhs: &[$ty], out: &mut [$ty]) {
            let mut index = 0;
            while index + $lanes <= lhs.len() {
                unsafe {
                    let left = $load(lhs.as_ptr().add(index));
                    let right = $load(rhs.as_ptr().add(index));
                    $store(out.as_mut_ptr().add(index), $op(left, right));
                }
                index += $lanes;
            }
            for index in index..lhs.len() {
                out[index] = lhs[index] $tail rhs[index];
            }
        }
    };
}

binary_kernel!(add_f32, f32, 4, vld1q_f32, vst1q_f32, vaddq_f32, +);
binary_kernel!(sub_f32, f32, 4, vld1q_f32, vst1q_f32, vsubq_f32, -);
binary_kernel!(mul_f32, f32, 4, vld1q_f32, vst1q_f32, vmulq_f32, *);
binary_kernel!(div_f32, f32, 4, vld1q_f32, vst1q_f32, vdivq_f32, /);
binary_kernel!(add_f64, f64, 2, vld1q_f64, vst1q_f64, vaddq_f64, +);
binary_kernel!(sub_f64, f64, 2, vld1q_f64, vst1q_f64, vsubq_f64, -);
binary_kernel!(mul_f64, f64, 2, vld1q_f64, vst1q_f64, vmulq_f64, *);
binary_kernel!(div_f64, f64, 2, vld1q_f64, vst1q_f64, vdivq_f64, /);

macro_rules! scalar_kernel {
    ($name:ident, $ty:ty, $lanes:expr, $load:ident, $store:ident, $dup:ident, $op:ident, $tail:tt) => {
        pub(crate) fn $name(input: &[$ty], scalar: $ty, out: &mut [$ty]) {
            let scalar_vector = unsafe { $dup(scalar) };
            let mut index = 0;
            while index + $lanes <= input.len() {
                unsafe {
                    let values = $load(input.as_ptr().add(index));
                    $store(out.as_mut_ptr().add(index), $op(values, scalar_vector));
                }
                index += $lanes;
            }
            for index in index..input.len() {
                out[index] = input[index] $tail scalar;
            }
        }
    };
}

scalar_kernel!(add_scalar_f32, f32, 4, vld1q_f32, vst1q_f32, vdupq_n_f32, vaddq_f32, +);
scalar_kernel!(sub_scalar_f32, f32, 4, vld1q_f32, vst1q_f32, vdupq_n_f32, vsubq_f32, -);
scalar_kernel!(mul_scalar_f32, f32, 4, vld1q_f32, vst1q_f32, vdupq_n_f32, vmulq_f32, *);
scalar_kernel!(div_scalar_f32, f32, 4, vld1q_f32, vst1q_f32, vdupq_n_f32, vdivq_f32, /);
scalar_kernel!(add_scalar_f64, f64, 2, vld1q_f64, vst1q_f64, vdupq_n_f64, vaddq_f64, +);
scalar_kernel!(sub_scalar_f64, f64, 2, vld1q_f64, vst1q_f64, vdupq_n_f64, vsubq_f64, -);
scalar_kernel!(mul_scalar_f64, f64, 2, vld1q_f64, vst1q_f64, vdupq_n_f64, vmulq_f64, *);
scalar_kernel!(div_scalar_f64, f64, 2, vld1q_f64, vst1q_f64, vdupq_n_f64, vdivq_f64, /);

macro_rules! unary_kernel {
    ($name:ident, $ty:ty, $lanes:expr, $load:ident, $store:ident, $op:ident, $tail:expr) => {
        pub(crate) fn $name(input: &[$ty], out: &mut [$ty]) {
            let mut index = 0;
            while index + $lanes <= input.len() {
                unsafe {
                    let values = $load(input.as_ptr().add(index));
                    $store(out.as_mut_ptr().add(index), $op(values));
                }
                index += $lanes;
            }
            for index in index..input.len() {
                out[index] = $tail(input[index]);
            }
        }
    };
}

unary_kernel!(neg_f32, f32, 4, vld1q_f32, vst1q_f32, vnegq_f32, |value: f32| -value);
unary_kernel!(abs_f32, f32, 4, vld1q_f32, vst1q_f32, vabsq_f32, f32::abs);
unary_kernel!(neg_f64, f64, 2, vld1q_f64, vst1q_f64, vnegq_f64, |value: f64| -value);
unary_kernel!(abs_f64, f64, 2, vld1q_f64, vst1q_f64, vabsq_f64, f64::abs);

pub(crate) fn sum_f32(values: &[f32]) -> f32 {
    let mut accumulator = unsafe { vdupq_n_f32(0.0) };
    let mut index = 0;
    while index + 4 <= values.len() {
        unsafe { accumulator = vaddq_f32(accumulator, vld1q_f32(values.as_ptr().add(index))) };
        index += 4;
    }
    let mut total = unsafe { vaddvq_f32(accumulator) };
    for &value in &values[index..] {
        total += value;
    }
    total
}

pub(crate) fn sum_f64(values: &[f64]) -> f64 {
    let mut accumulator = unsafe { vdupq_n_f64(0.0) };
    let mut index = 0;
    while index + 2 <= values.len() {
        unsafe { accumulator = vaddq_f64(accumulator, vld1q_f64(values.as_ptr().add(index))) };
        index += 2;
    }
    let mut total = unsafe { vaddvq_f64(accumulator) };
    for &value in &values[index..] {
        total += value;
    }
    total
}

pub(crate) fn all_bool(values: &[bool]) -> bool {
    let mut index = 0;
    while index + 16 <= values.len() {
        let lanes = unsafe { vld1q_u8(values.as_ptr().add(index).cast()) };
        if unsafe { vminvq_u8(lanes) } == 0 {
            return false;
        }
        index += 16;
    }
    values[index..].iter().all(|&value| value)
}

pub(crate) fn any_bool(values: &[bool]) -> bool {
    let mut index = 0;
    while index + 16 <= values.len() {
        let lanes = unsafe { vld1q_u8(values.as_ptr().add(index).cast()) };
        if unsafe { vmaxvq_u8(lanes) } != 0 {
            return true;
        }
        index += 16;
    }
    values[index..].iter().any(|&value| value)
}

pub(crate) fn count_true(values: &[bool]) -> usize {
    let mut total = 0;
    let mut index = 0;
    while index + 16 <= values.len() {
        let lanes = unsafe { vld1q_u8(values.as_ptr().add(index).cast()) };
        total += usize::from(unsafe { vaddvq_u8(lanes) });
        index += 16;
    }
    total + values[index..].iter().filter(|&&value| value).count()
}

pub(crate) fn squared_deviations_f32(values: &[f32], mean: f64) -> f64 {
    let mean_vector = unsafe { vdupq_n_f64(mean) };
    let mut accumulator_low = unsafe { vdupq_n_f64(0.0) };
    let mut accumulator_high = unsafe { vdupq_n_f64(0.0) };
    let mut index = 0;
    while index + 4 <= values.len() {
        unsafe {
            let lanes = vld1q_f32(values.as_ptr().add(index));
            let difference_low = vsubq_f64(vcvt_f64_f32(vget_low_f32(lanes)), mean_vector);
            let difference_high = vsubq_f64(vcvt_f64_f32(vget_high_f32(lanes)), mean_vector);
            accumulator_low = vfmaq_f64(accumulator_low, difference_low, difference_low);
            accumulator_high = vfmaq_f64(accumulator_high, difference_high, difference_high);
        }
        index += 4;
    }
    let mut total = unsafe { vaddvq_f64(vaddq_f64(accumulator_low, accumulator_high)) };
    for &value in &values[index..] {
        let difference = f64::from(value) - mean;
        total += difference * difference;
    }
    total
}

pub(crate) fn squared_deviations_f64(values: &[f64], mean: f64) -> f64 {
    let mean_vector = unsafe { vdupq_n_f64(mean) };
    let mut accumulator = unsafe { vdupq_n_f64(0.0) };
    let mut index = 0;
    while index + 2 <= values.len() {
        unsafe {
            let difference = vsubq_f64(vld1q_f64(values.as_ptr().add(index)), mean_vector);
            accumulator = vfmaq_f64(accumulator, difference, difference);
        }
        index += 2;
    }
    let mut total = unsafe { vaddvq_f64(accumulator) };
    for &value in &values[index..] {
        let difference = value - mean;
        total += difference * difference;
    }
    total
}

pub(crate) fn prod_f32(values: &[f32]) -> f32 {
    let mut accumulator = unsafe { vdupq_n_f32(1.0) };
    let mut index = 0;
    while index + 4 <= values.len() {
        unsafe { accumulator = vmulq_f32(accumulator, vld1q_f32(values.as_ptr().add(index))) };
        index += 4;
    }
    let mut lanes = [1.0; 4];
    unsafe { vst1q_f32(lanes.as_mut_ptr(), accumulator) };
    let mut total = lanes.into_iter().product();
    for &value in &values[index..] {
        total *= value;
    }
    total
}

pub(crate) fn prod_f64(values: &[f64]) -> f64 {
    let mut accumulator = unsafe { vdupq_n_f64(1.0) };
    let mut index = 0;
    while index + 2 <= values.len() {
        unsafe { accumulator = vmulq_f64(accumulator, vld1q_f64(values.as_ptr().add(index))) };
        index += 2;
    }
    let mut lanes = [1.0; 2];
    unsafe { vst1q_f64(lanes.as_mut_ptr(), accumulator) };
    let mut total = lanes.into_iter().product();
    for &value in &values[index..] {
        total *= value;
    }
    total
}

pub(crate) fn min_f32(values: &[f32]) -> f32 {
    if values.len() < 4 {
        return values.iter().copied().fold(f32::INFINITY, f32::min);
    }
    let mut accumulator = unsafe { vld1q_f32(values.as_ptr()) };
    let mut index = 4;
    while index + 4 <= values.len() {
        unsafe { accumulator = vminq_f32(accumulator, vld1q_f32(values.as_ptr().add(index))) };
        index += 4;
    }
    let mut lanes = [f32::INFINITY; 4];
    unsafe { vst1q_f32(lanes.as_mut_ptr(), accumulator) };
    let mut minimum = lanes.into_iter().fold(f32::INFINITY, f32::min);
    for &value in &values[index..] {
        minimum = minimum.min(value);
    }
    minimum
}

pub(crate) fn min_f64(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return values.iter().copied().fold(f64::INFINITY, f64::min);
    }
    let mut accumulator = unsafe { vld1q_f64(values.as_ptr()) };
    let mut index = 2;
    while index + 2 <= values.len() {
        unsafe { accumulator = vminq_f64(accumulator, vld1q_f64(values.as_ptr().add(index))) };
        index += 2;
    }
    let mut lanes = [f64::INFINITY; 2];
    unsafe { vst1q_f64(lanes.as_mut_ptr(), accumulator) };
    let mut minimum = lanes.into_iter().fold(f64::INFINITY, f64::min);
    for &value in &values[index..] {
        minimum = minimum.min(value);
    }
    minimum
}

pub(crate) fn max_f32(values: &[f32]) -> f32 {
    if values.len() < 4 {
        return values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    }
    let mut accumulator = unsafe { vld1q_f32(values.as_ptr()) };
    let mut index = 4;
    while index + 4 <= values.len() {
        unsafe { accumulator = vmaxq_f32(accumulator, vld1q_f32(values.as_ptr().add(index))) };
        index += 4;
    }
    let mut lanes = [f32::NEG_INFINITY; 4];
    unsafe { vst1q_f32(lanes.as_mut_ptr(), accumulator) };
    let mut maximum = lanes.into_iter().fold(f32::NEG_INFINITY, f32::max);
    for &value in &values[index..] {
        maximum = maximum.max(value);
    }
    maximum
}

pub(crate) fn max_f64(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    }
    let mut accumulator = unsafe { vld1q_f64(values.as_ptr()) };
    let mut index = 2;
    while index + 2 <= values.len() {
        unsafe { accumulator = vmaxq_f64(accumulator, vld1q_f64(values.as_ptr().add(index))) };
        index += 2;
    }
    let mut lanes = [f64::NEG_INFINITY; 2];
    unsafe { vst1q_f64(lanes.as_mut_ptr(), accumulator) };
    let mut maximum = lanes.into_iter().fold(f64::NEG_INFINITY, f64::max);
    for &value in &values[index..] {
        maximum = maximum.max(value);
    }
    maximum
}
