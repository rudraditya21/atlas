use std::{
    any::TypeId,
    mem::{align_of, size_of},
};

use atlas_ndarray::Numeric;

const F32_LANES: usize = 8;
const F64_LANES: usize = 4;

pub(crate) fn dot_contiguous<T: Numeric>(lhs: &[T], rhs: &[T]) -> T {
    debug_assert_eq!(lhs.len(), rhs.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_f32::<T>() && std::is_x86_feature_detected!("avx") {
            // SAFETY: The runtime check guarantees AVX support and the type check guarantees
            // identical element layouts for the casted slices.
            let value = unsafe { x86_64::dot_f32(cast_slice(lhs), cast_slice(rhs)) };
            return cast_value(value);
        }

        if is_f64::<T>() && std::is_x86_feature_detected!("avx") {
            // SAFETY: The runtime check guarantees AVX support and the type check guarantees
            // identical element layouts for the casted slices.
            let value = unsafe { x86_64::dot_f64(cast_slice(lhs), cast_slice(rhs)) };
            return cast_value(value);
        }
    }

    dot_scalar(lhs, rhs)
}

pub(crate) fn scaled_accumulate_contiguous<T: Numeric>(output: &mut [T], input: &[T], scale: T) {
    debug_assert_eq!(output.len(), input.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_f32::<T>() && std::is_x86_feature_detected!("avx") {
            // SAFETY: The runtime check guarantees AVX support and the type check guarantees
            // identical element layouts for the casted slices and scalar.
            unsafe {
                x86_64::scaled_accumulate_f32(
                    cast_mut_slice(output),
                    cast_slice(input),
                    cast_value(scale),
                );
            }
            return;
        }

        if is_f64::<T>() && std::is_x86_feature_detected!("avx") {
            // SAFETY: The runtime check guarantees AVX support and the type check guarantees
            // identical element layouts for the casted slices and scalar.
            unsafe {
                x86_64::scaled_accumulate_f64(
                    cast_mut_slice(output),
                    cast_slice(input),
                    cast_value(scale),
                );
            }
            return;
        }
    }

    scaled_accumulate_scalar(output, input, scale);
}

fn dot_scalar<T: Numeric>(lhs: &[T], rhs: &[T]) -> T {
    let mut total = T::zero();

    for (&left, &right) in lhs.iter().zip(rhs) {
        total += left * right;
    }

    total
}

fn scaled_accumulate_scalar<T: Numeric>(output: &mut [T], input: &[T], scale: T) {
    for (dst, &src) in output.iter_mut().zip(input) {
        *dst += scale * src;
    }
}

#[inline]
pub(crate) fn is_f32<T: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

#[inline]
pub(crate) fn is_f64<T: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f64>()
}

#[inline]
pub(crate) fn cast_value<U, T>(value: U) -> T
where
    U: Copy + 'static,
    T: Copy + 'static,
{
    assert_exact_type::<U, T>();
    // SAFETY: Callers only use this after an exact type match between U and T.
    unsafe { std::mem::transmute_copy::<U, T>(&value) }
}

#[inline]
pub(crate) fn cast_slice<T: 'static, U: 'static>(data: &[T]) -> &[U] {
    assert_exact_type::<T, U>();
    // SAFETY: Callers only use this after an exact type match between T and U.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const U, data.len()) }
}

#[inline]
pub(crate) fn cast_mut_slice<T: 'static, U: 'static>(data: &mut [T]) -> &mut [U] {
    assert_exact_type::<T, U>();
    // SAFETY: Callers only use this after an exact type match between T and U.
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut U, data.len()) }
}

#[inline]
fn assert_exact_type<T: 'static, U: 'static>() {
    assert_eq!(TypeId::of::<T>(), TypeId::of::<U>());
    assert_eq!(size_of::<T>(), size_of::<U>());
    assert_eq!(align_of::<T>(), align_of::<U>());
}

pub(crate) fn dot_contiguous_f32(lhs: &[f32], rhs: &[f32]) -> f32 {
    debug_assert_eq!(lhs.len(), rhs.len());

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: AVX support is verified at runtime.
            return unsafe { x86_64::dot_f32(lhs, rhs) };
        }
    }

    lhs.iter().zip(rhs).map(|(&left, &right)| left * right).sum()
}

pub(crate) fn dot_contiguous_f64(lhs: &[f64], rhs: &[f64]) -> f64 {
    debug_assert_eq!(lhs.len(), rhs.len());

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: AVX support is verified at runtime.
            return unsafe { x86_64::dot_f64(lhs, rhs) };
        }
    }

    lhs.iter().zip(rhs).map(|(&left, &right)| left * right).sum()
}

pub(crate) fn scaled_accumulate_contiguous_f32(output: &mut [f32], input: &[f32], scale: f32) {
    debug_assert_eq!(output.len(), input.len());

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: AVX support is verified at runtime.
            unsafe {
                x86_64::scaled_accumulate_f32(output, input, scale);
            }
            return;
        }
    }

    for (dst, &src) in output.iter_mut().zip(input) {
        *dst += scale * src;
    }
}

pub(crate) fn scaled_accumulate_contiguous_f64(output: &mut [f64], input: &[f64], scale: f64) {
    debug_assert_eq!(output.len(), input.len());

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: AVX support is verified at runtime.
            unsafe {
                x86_64::scaled_accumulate_f64(output, input, scale);
            }
            return;
        }
    }

    for (dst, &src) in output.iter_mut().zip(input) {
        *dst += scale * src;
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use std::arch::x86_64::*;

    use super::{F32_LANES, F64_LANES};

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn dot_f32(lhs: &[f32], rhs: &[f32]) -> f32 {
        if lhs.len() < F32_LANES {
            return lhs.iter().zip(rhs).map(|(&left, &right)| left * right).sum();
        }

        let body_len = lhs.len() / F32_LANES * F32_LANES;
        let mut acc = _mm256_setzero_ps();
        let mut index = 0;

        while index < body_len {
            let left = _mm256_loadu_ps(lhs.as_ptr().add(index));
            let right = _mm256_loadu_ps(rhs.as_ptr().add(index));
            let product = _mm256_mul_ps(left, right);
            acc = _mm256_add_ps(acc, product);
            index += F32_LANES;
        }

        let mut lanes = [0.0_f32; F32_LANES];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        let mut total: f32 = lanes.into_iter().sum();

        while index < lhs.len() {
            total += lhs[index] * rhs[index];
            index += 1;
        }

        total
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn dot_f64(lhs: &[f64], rhs: &[f64]) -> f64 {
        if lhs.len() < F64_LANES {
            return lhs.iter().zip(rhs).map(|(&left, &right)| left * right).sum();
        }

        let body_len = lhs.len() / F64_LANES * F64_LANES;
        let mut acc = _mm256_setzero_pd();
        let mut index = 0;

        while index < body_len {
            let left = _mm256_loadu_pd(lhs.as_ptr().add(index));
            let right = _mm256_loadu_pd(rhs.as_ptr().add(index));
            let product = _mm256_mul_pd(left, right);
            acc = _mm256_add_pd(acc, product);
            index += F64_LANES;
        }

        let mut lanes = [0.0_f64; F64_LANES];
        _mm256_storeu_pd(lanes.as_mut_ptr(), acc);
        let mut total: f64 = lanes.into_iter().sum();

        while index < lhs.len() {
            total += lhs[index] * rhs[index];
            index += 1;
        }

        total
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn scaled_accumulate_f32(output: &mut [f32], input: &[f32], scale: f32) {
        let body_len = output.len() / F32_LANES * F32_LANES;
        let scale_vector = _mm256_set1_ps(scale);
        let mut index = 0;

        while index < body_len {
            let dst = _mm256_loadu_ps(output.as_ptr().add(index));
            let src = _mm256_loadu_ps(input.as_ptr().add(index));
            let product = _mm256_mul_ps(scale_vector, src);
            let result = _mm256_add_ps(dst, product);
            _mm256_storeu_ps(output.as_mut_ptr().add(index), result);
            index += F32_LANES;
        }

        while index < output.len() {
            output[index] += scale * input[index];
            index += 1;
        }
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn scaled_accumulate_f64(output: &mut [f64], input: &[f64], scale: f64) {
        let body_len = output.len() / F64_LANES * F64_LANES;
        let scale_vector = _mm256_set1_pd(scale);
        let mut index = 0;

        while index < body_len {
            let dst = _mm256_loadu_pd(output.as_ptr().add(index));
            let src = _mm256_loadu_pd(input.as_ptr().add(index));
            let product = _mm256_mul_pd(scale_vector, src);
            let result = _mm256_add_pd(dst, product);
            _mm256_storeu_pd(output.as_mut_ptr().add(index), result);
            index += F64_LANES;
        }

        while index < output.len() {
            output[index] += scale * input[index];
            index += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{dot_contiguous, scaled_accumulate_contiguous};

    #[test]
    fn contiguous_dot_matches_expected_results() {
        let lhs = vec![1.0_f32; 19];
        let rhs = vec![2.0_f32; 19];

        assert_eq!(dot_contiguous(&lhs, &rhs), 38.0);
        assert_eq!(dot_contiguous(&[1_i32, 2, 3], &[4_i32, 5, 6]), 32);
    }

    #[test]
    fn scaled_accumulate_matches_expected_results() {
        let mut output = vec![1.0_f64; 10];
        let input = vec![2.0_f64; 10];

        scaled_accumulate_contiguous(&mut output, &input, 3.0);
        assert!(output.iter().all(|&value| value == 7.0));

        let mut integer_output = vec![1_i32, 2, 3];
        scaled_accumulate_contiguous(&mut integer_output, &[4_i32, 5, 6], 2);
        assert_eq!(integer_output, vec![9, 12, 15]);
    }
}
