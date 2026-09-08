use num_traits::ToPrimitive;

use crate::{
    AtlasNdError, AtlasNdResult, ElementwiseArithmetic, Numeric,
    simd_support::{cast_mut_slice, cast_value},
};

pub(crate) use crate::simd_support::{cast_slice, is_f32, is_f64};

const SIMD_LANES: usize = 8;
const SIMD_REDUCTION_THRESHOLD: usize = SIMD_LANES * 32;

pub(crate) fn add_contiguous<T: ElementwiseArithmetic>(lhs: &[T], rhs: &[T], out: &mut [T]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    debug_assert_eq!(lhs.len(), out.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_f32::<T>() && std::is_x86_feature_detected!("avx") {
            // SAFETY: The type check guarantees exact element layout.
            unsafe {
                x86_64::add_f32(cast_slice(lhs), cast_slice(rhs), cast_mut_slice(out));
            }
            return;
        }

        if is_f64::<T>() && std::is_x86_feature_detected!("avx") {
            // SAFETY: The type check guarantees exact element layout.
            unsafe {
                x86_64::add_f64(cast_slice(lhs), cast_slice(rhs), cast_mut_slice(out));
            }
            return;
        }
    }

    map_binary_scalar(lhs, rhs, out, ElementwiseArithmetic::elementwise_add);
}

pub(crate) fn mul_contiguous<T: ElementwiseArithmetic>(lhs: &[T], rhs: &[T], out: &mut [T]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    debug_assert_eq!(lhs.len(), out.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_f32::<T>() && std::is_x86_feature_detected!("avx") {
            // SAFETY: The type check guarantees exact element layout.
            unsafe {
                x86_64::mul_f32(cast_slice(lhs), cast_slice(rhs), cast_mut_slice(out));
            }
            return;
        }

        if is_f64::<T>() && std::is_x86_feature_detected!("avx") {
            // SAFETY: The type check guarantees exact element layout.
            unsafe {
                x86_64::mul_f64(cast_slice(lhs), cast_slice(rhs), cast_mut_slice(out));
            }
            return;
        }
    }

    map_binary_scalar(lhs, rhs, out, ElementwiseArithmetic::elementwise_mul);
}

pub(crate) fn add_scalar_contiguous<T: ElementwiseArithmetic>(
    input: &[T],
    scalar: T,
    out: &mut [T],
) {
    debug_assert_eq!(input.len(), out.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_f32::<T>() && std::is_x86_feature_detected!("avx") {
            // SAFETY: The type check guarantees exact element layout.
            unsafe {
                x86_64::add_scalar_f32(cast_slice(input), to_f32(scalar), cast_mut_slice(out));
            }
            return;
        }

        if is_f64::<T>() && std::is_x86_feature_detected!("avx") {
            // SAFETY: The type check guarantees exact element layout.
            unsafe {
                x86_64::add_scalar_f64(cast_slice(input), to_f64(scalar), cast_mut_slice(out));
            }
            return;
        }
    }

    map_scalar_scalar(input, scalar, out, ElementwiseArithmetic::elementwise_add);
}

pub(crate) fn mul_scalar_contiguous<T: ElementwiseArithmetic>(
    input: &[T],
    scalar: T,
    out: &mut [T],
) {
    debug_assert_eq!(input.len(), out.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_f32::<T>() && std::is_x86_feature_detected!("avx") {
            // SAFETY: The type check guarantees exact element layout.
            unsafe {
                x86_64::mul_scalar_f32(cast_slice(input), to_f32(scalar), cast_mut_slice(out));
            }
            return;
        }

        if is_f64::<T>() && std::is_x86_feature_detected!("avx") {
            // SAFETY: The type check guarantees exact element layout.
            unsafe {
                x86_64::mul_scalar_f64(cast_slice(input), to_f64(scalar), cast_mut_slice(out));
            }
            return;
        }
    }

    map_scalar_scalar(input, scalar, out, ElementwiseArithmetic::elementwise_mul);
}

pub(crate) fn map_binary_contiguous<T, F>(lhs: &[T], rhs: &[T], out: &mut [T], op: F)
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
{
    map_binary_scalar(lhs, rhs, out, op);
}

pub(crate) fn map_scalar_contiguous<T, F>(input: &[T], scalar: T, out: &mut [T], op: F)
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
{
    map_scalar_scalar(input, scalar, out, op);
}

pub(crate) fn sum_contiguous<T: ElementwiseArithmetic>(values: &[T]) -> T {
    #[cfg(target_arch = "x86_64")]
    {
        if values.len() >= SIMD_REDUCTION_THRESHOLD
            && is_f32::<T>()
            && std::is_x86_feature_detected!("avx")
        {
            // SAFETY: The type check guarantees exact element layout.
            return cast_value_exact(unsafe { x86_64::sum_f32(cast_slice(values)) });
        }

        if values.len() >= SIMD_REDUCTION_THRESHOLD
            && is_f64::<T>()
            && std::is_x86_feature_detected!("avx")
        {
            // SAFETY: The type check guarantees exact element layout.
            return cast_value_exact(unsafe { x86_64::sum_f64(cast_slice(values)) });
        }
    }

    if is_f32::<T>() {
        return cast_value_exact(compensated_sum_f32(cast_slice(values)));
    }

    if is_f64::<T>() {
        return cast_value_exact(compensated_sum_f64(cast_slice(values)));
    }

    sum_scalar(values)
}

pub(crate) fn prod_contiguous<T: ElementwiseArithmetic>(values: &[T]) -> T {
    #[cfg(target_arch = "x86_64")]
    {
        if is_f32::<T>() && std::is_x86_feature_detected!("avx") {
            // SAFETY: The type check guarantees exact element layout.
            let value = unsafe { x86_64::prod_f32(cast_slice(values)) };
            return cast_value_exact(value);
        }

        if is_f64::<T>() && std::is_x86_feature_detected!("avx") {
            // SAFETY: The type check guarantees exact element layout.
            let value = unsafe { x86_64::prod_f64(cast_slice(values)) };
            return cast_value_exact(value);
        }
    }

    prod_scalar(values)
}

pub(crate) fn min_contiguous<T>(values: &[T], op: &'static str) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    if let Some(value) = first_unordered(values) {
        return Ok(value);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_f32::<T>() && std::is_x86_feature_detected!("avx") {
            if values.is_empty() {
                return Err(AtlasNdError::EmptyReduction { op });
            }

            // SAFETY: The type check guarantees exact element layout.
            let value = unsafe { x86_64::min_f32(cast_slice(values)) };
            return Ok(cast_value_exact(value));
        }

        if is_f64::<T>() && std::is_x86_feature_detected!("avx") {
            if values.is_empty() {
                return Err(AtlasNdError::EmptyReduction { op });
            }

            // SAFETY: The type check guarantees exact element layout.
            let value = unsafe { x86_64::min_f64(cast_slice(values)) };
            return Ok(cast_value_exact(value));
        }
    }

    min_scalar(values, op)
}

pub(crate) fn max_contiguous<T>(values: &[T], op: &'static str) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    if let Some(value) = first_unordered(values) {
        return Ok(value);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_f32::<T>() && std::is_x86_feature_detected!("avx") {
            if values.is_empty() {
                return Err(AtlasNdError::EmptyReduction { op });
            }

            // SAFETY: The type check guarantees exact element layout.
            let value = unsafe { x86_64::max_f32(cast_slice(values)) };
            return Ok(cast_value_exact(value));
        }

        if is_f64::<T>() && std::is_x86_feature_detected!("avx") {
            if values.is_empty() {
                return Err(AtlasNdError::EmptyReduction { op });
            }

            // SAFETY: The type check guarantees exact element layout.
            let value = unsafe { x86_64::max_f64(cast_slice(values)) };
            return Ok(cast_value_exact(value));
        }
    }

    max_scalar(values, op)
}

pub(crate) fn mean_contiguous<T>(values: &[T], op: &'static str) -> AtlasNdResult<f64>
where
    T: Numeric + ToPrimitive,
{
    if values.is_empty() {
        return Err(AtlasNdError::EmptyReduction { op });
    }

    if is_f32::<T>() {
        #[cfg(target_arch = "x86_64")]
        if values.len() >= SIMD_REDUCTION_THRESHOLD && std::is_x86_feature_detected!("avx") {
            // SAFETY: The type check guarantees exact element layout.
            return Ok(unsafe { x86_64::sum_f32(cast_slice(values)) } as f64 / values.len() as f64);
        }
        return Ok(compensated_sum_f32_as_f64(cast_slice(values)) / values.len() as f64);
    }

    if is_f64::<T>() {
        #[cfg(target_arch = "x86_64")]
        if values.len() >= SIMD_REDUCTION_THRESHOLD && std::is_x86_feature_detected!("avx") {
            // SAFETY: The type check guarantees exact element layout.
            return Ok(unsafe { x86_64::sum_f64(cast_slice(values)) } / values.len() as f64);
        }
        return Ok(compensated_sum_f64(cast_slice(values)) / values.len() as f64);
    }

    mean_scalar(values, op)
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    pub(crate) const fn new() -> Self {
        Self { sum: 0.0, correction: 0.0 }
    }

    pub(crate) fn add(&mut self, value: f64) {
        let adjusted = self.sum + value;

        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - adjusted) + value;
        } else {
            self.correction += (value - adjusted) + self.sum;
        }

        self.sum = adjusted;
    }

    pub(crate) fn finish(self) -> f64 {
        self.sum + self.correction
    }
}

fn map_binary_scalar<T, F>(lhs: &[T], rhs: &[T], out: &mut [T], op: F)
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
{
    let len = out.len();
    let body_len = body_len(len);
    let mut index = 0;

    while index < body_len {
        out[index] = op(lhs[index], rhs[index]);
        out[index + 1] = op(lhs[index + 1], rhs[index + 1]);
        out[index + 2] = op(lhs[index + 2], rhs[index + 2]);
        out[index + 3] = op(lhs[index + 3], rhs[index + 3]);
        out[index + 4] = op(lhs[index + 4], rhs[index + 4]);
        out[index + 5] = op(lhs[index + 5], rhs[index + 5]);
        out[index + 6] = op(lhs[index + 6], rhs[index + 6]);
        out[index + 7] = op(lhs[index + 7], rhs[index + 7]);
        index += SIMD_LANES;
    }

    while index < len {
        out[index] = op(lhs[index], rhs[index]);
        index += 1;
    }
}

fn map_scalar_scalar<T, F>(input: &[T], scalar: T, out: &mut [T], op: F)
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
{
    let len = out.len();
    let body_len = body_len(len);
    let mut index = 0;

    while index < body_len {
        out[index] = op(input[index], scalar);
        out[index + 1] = op(input[index + 1], scalar);
        out[index + 2] = op(input[index + 2], scalar);
        out[index + 3] = op(input[index + 3], scalar);
        out[index + 4] = op(input[index + 4], scalar);
        out[index + 5] = op(input[index + 5], scalar);
        out[index + 6] = op(input[index + 6], scalar);
        out[index + 7] = op(input[index + 7], scalar);
        index += SIMD_LANES;
    }

    while index < len {
        out[index] = op(input[index], scalar);
        index += 1;
    }
}

fn sum_scalar<T: ElementwiseArithmetic>(values: &[T]) -> T {
    let len = values.len();
    let body_len = body_len(len);
    let mut acc0 = T::zero();
    let mut acc1 = T::zero();
    let mut acc2 = T::zero();
    let mut acc3 = T::zero();
    let mut acc4 = T::zero();
    let mut acc5 = T::zero();
    let mut acc6 = T::zero();
    let mut acc7 = T::zero();
    let mut index = 0;

    while index < body_len {
        acc0 = acc0.elementwise_add(values[index]);
        acc1 = acc1.elementwise_add(values[index + 1]);
        acc2 = acc2.elementwise_add(values[index + 2]);
        acc3 = acc3.elementwise_add(values[index + 3]);
        acc4 = acc4.elementwise_add(values[index + 4]);
        acc5 = acc5.elementwise_add(values[index + 5]);
        acc6 = acc6.elementwise_add(values[index + 6]);
        acc7 = acc7.elementwise_add(values[index + 7]);
        index += SIMD_LANES;
    }

    let mut total = acc0.elementwise_add(acc1);
    total = total.elementwise_add(acc2.elementwise_add(acc3));
    total = total.elementwise_add(acc4.elementwise_add(acc5));
    total = total.elementwise_add(acc6.elementwise_add(acc7));

    while index < len {
        total = total.elementwise_add(values[index]);
        index += 1;
    }

    total
}

fn prod_scalar<T: ElementwiseArithmetic>(values: &[T]) -> T {
    let len = values.len();
    let body_len = body_len(len);
    let mut acc0 = T::one();
    let mut acc1 = T::one();
    let mut acc2 = T::one();
    let mut acc3 = T::one();
    let mut acc4 = T::one();
    let mut acc5 = T::one();
    let mut acc6 = T::one();
    let mut acc7 = T::one();
    let mut index = 0;

    while index < body_len {
        acc0 = acc0.elementwise_mul(values[index]);
        acc1 = acc1.elementwise_mul(values[index + 1]);
        acc2 = acc2.elementwise_mul(values[index + 2]);
        acc3 = acc3.elementwise_mul(values[index + 3]);
        acc4 = acc4.elementwise_mul(values[index + 4]);
        acc5 = acc5.elementwise_mul(values[index + 5]);
        acc6 = acc6.elementwise_mul(values[index + 6]);
        acc7 = acc7.elementwise_mul(values[index + 7]);
        index += SIMD_LANES;
    }

    let mut total = acc0.elementwise_mul(acc1);
    total = total.elementwise_mul(acc2.elementwise_mul(acc3));
    total = total.elementwise_mul(acc4.elementwise_mul(acc5));
    total = total.elementwise_mul(acc6.elementwise_mul(acc7));

    while index < len {
        total = total.elementwise_mul(values[index]);
        index += 1;
    }

    total
}

fn min_scalar<T>(values: &[T], op: &'static str) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    let mut iter = values.iter().copied();
    let mut minimum = iter.next().ok_or(AtlasNdError::EmptyReduction { op })?;

    for value in iter {
        minimum = min_propagating(minimum, value);
    }

    Ok(minimum)
}

fn max_scalar<T>(values: &[T], op: &'static str) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    let mut iter = values.iter().copied();
    let mut maximum = iter.next().ok_or(AtlasNdError::EmptyReduction { op })?;

    for value in iter {
        maximum = max_propagating(maximum, value);
    }

    Ok(maximum)
}

pub(crate) fn min_propagating<T: Numeric + PartialOrd>(current: T, value: T) -> T {
    if current.partial_cmp(&current).is_none() {
        current
    } else if value.partial_cmp(&value).is_none() || value < current {
        value
    } else {
        current
    }
}

pub(crate) fn max_propagating<T: Numeric + PartialOrd>(current: T, value: T) -> T {
    if current.partial_cmp(&current).is_none() {
        current
    } else if value.partial_cmp(&value).is_none() || value > current {
        value
    } else {
        current
    }
}

fn first_unordered<T: Copy + PartialOrd>(values: &[T]) -> Option<T> {
    values.iter().copied().find(|value| value.partial_cmp(value).is_none())
}

fn mean_scalar<T>(values: &[T], op: &'static str) -> AtlasNdResult<f64>
where
    T: Numeric + ToPrimitive,
{
    let mut total = CompensatedSum::new();

    for &value in values {
        total.add(value.to_f64().ok_or(AtlasNdError::NumericConversionFailed { op })?);
    }

    Ok(total.finish() / values.len() as f64)
}

#[inline]
fn body_len(len: usize) -> usize {
    len / SIMD_LANES * SIMD_LANES
}

#[inline]
fn to_f32<T: Numeric>(value: T) -> f32 {
    cast_value(value)
}

#[inline]
fn to_f64<T: Numeric>(value: T) -> f64 {
    cast_value(value)
}

#[inline]
pub(crate) fn cast_value_exact<U, T>(value: U) -> T
where
    U: Copy + 'static,
    T: Copy + 'static,
{
    cast_value(value)
}

fn compensated_sum_f32(values: &[f32]) -> f32 {
    compensated_sum_f32_as_f64(values) as f32
}

pub(crate) fn compensated_sum_f32_as_f64(values: &[f32]) -> f64 {
    let mut total = CompensatedSum::new();

    for &value in values {
        total.add(f64::from(value));
    }

    total.finish()
}

pub(crate) fn compensated_sum_f64(values: &[f64]) -> f64 {
    let mut total = CompensatedSum::new();

    for &value in values {
        total.add(value);
    }

    total.finish()
}

pub(crate) fn compensated_sum_strided_f32(
    values: &[f32],
    offset: usize,
    len: usize,
    stride: usize,
) -> f32 {
    compensated_sum_strided_f32_as_f64(values, offset, len, stride) as f32
}

pub(crate) fn compensated_sum_strided_f32_as_f64(
    values: &[f32],
    offset: usize,
    len: usize,
    stride: usize,
) -> f64 {
    let mut total = CompensatedSum::new();

    for index in 0..len {
        total.add(f64::from(values[offset + index * stride]));
    }

    total.finish()
}

pub(crate) fn compensated_sum_strided_f64(
    values: &[f64],
    offset: usize,
    len: usize,
    stride: usize,
) -> f64 {
    let mut total = CompensatedSum::new();

    for index in 0..len {
        total.add(values[offset + index * stride]);
    }

    total.finish()
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn add_f32(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
        // SAFETY: The caller verifies AVX support before invoking this kernel.
        unsafe { map_f32(lhs, rhs, out, _mm256_add_ps) };
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn mul_f32(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
        // SAFETY: The caller verifies AVX support before invoking this kernel.
        unsafe { map_f32(lhs, rhs, out, _mm256_mul_ps) };
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn add_scalar_f32(input: &[f32], scalar: f32, out: &mut [f32]) {
        // SAFETY: The caller verifies AVX support before invoking this kernel.
        unsafe { map_scalar_f32(input, scalar, out, _mm256_add_ps) };
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn mul_scalar_f32(input: &[f32], scalar: f32, out: &mut [f32]) {
        // SAFETY: The caller verifies AVX support before invoking this kernel.
        unsafe { map_scalar_f32(input, scalar, out, _mm256_mul_ps) };
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn add_f64(lhs: &[f64], rhs: &[f64], out: &mut [f64]) {
        // SAFETY: The caller verifies AVX support before invoking this kernel.
        unsafe { map_f64(lhs, rhs, out, _mm256_add_pd) };
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn mul_f64(lhs: &[f64], rhs: &[f64], out: &mut [f64]) {
        // SAFETY: The caller verifies AVX support before invoking this kernel.
        unsafe { map_f64(lhs, rhs, out, _mm256_mul_pd) };
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn add_scalar_f64(input: &[f64], scalar: f64, out: &mut [f64]) {
        // SAFETY: The caller verifies AVX support before invoking this kernel.
        unsafe { map_scalar_f64(input, scalar, out, _mm256_add_pd) };
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn mul_scalar_f64(input: &[f64], scalar: f64, out: &mut [f64]) {
        // SAFETY: The caller verifies AVX support before invoking this kernel.
        unsafe { map_scalar_f64(input, scalar, out, _mm256_mul_pd) };
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn sum_f32(values: &[f32]) -> f32 {
        let mut index = 0;
        // SAFETY: This function requires AVX support.
        let mut accumulator = { _mm256_setzero_ps() };

        while index + 8 <= values.len() {
            // SAFETY: `index + 8 <= values.len()` keeps the unaligned load in bounds.
            accumulator =
                unsafe { _mm256_add_ps(accumulator, _mm256_loadu_ps(values.as_ptr().add(index))) };
            index += 8;
        }

        let mut lanes = [0.0_f32; 8];
        // SAFETY: `lanes` holds exactly eight f32 values.
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), accumulator) };
        let mut total = super::CompensatedSum::new();
        for value in lanes {
            total.add(f64::from(value));
        }
        while index < values.len() {
            total.add(f64::from(values[index]));
            index += 1;
        }
        total.finish() as f32
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn sum_f64(values: &[f64]) -> f64 {
        let mut index = 0;
        // SAFETY: This function requires AVX support.
        let mut accumulator = { _mm256_setzero_pd() };

        while index + 4 <= values.len() {
            // SAFETY: `index + 4 <= values.len()` keeps the unaligned load in bounds.
            accumulator =
                unsafe { _mm256_add_pd(accumulator, _mm256_loadu_pd(values.as_ptr().add(index))) };
            index += 4;
        }

        let mut lanes = [0.0_f64; 4];
        // SAFETY: `lanes` holds exactly four f64 values.
        unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), accumulator) };
        let mut total = super::CompensatedSum::new();
        for value in lanes {
            total.add(value);
        }
        while index < values.len() {
            total.add(values[index]);
            index += 1;
        }
        total.finish()
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn prod_f32(values: &[f32]) -> f32 {
        if values.len() < 8 {
            return values.iter().copied().product();
        }

        let body_len = values.len() / 8 * 8;
        // SAFETY: This function requires AVX support.
        let mut acc = { _mm256_set1_ps(1.0) };
        let mut index = 0;

        while index < body_len {
            // SAFETY: `index < body_len` guarantees eight readable values.
            acc = unsafe { _mm256_mul_ps(acc, _mm256_loadu_ps(values.as_ptr().add(index))) };
            index += 8;
        }

        let mut lanes = [1.0_f32; 8];
        // SAFETY: `lanes` holds exactly eight f32 values.
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
        let mut total: f32 = lanes.into_iter().product();

        while index < values.len() {
            total *= values[index];
            index += 1;
        }

        total
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn min_f32(values: &[f32]) -> f32 {
        if values.len() < 8 {
            return values.iter().copied().fold(f32::INFINITY, f32::min);
        }

        let body_len = values.len() / 8 * 8;
        let mut index = 0;
        // SAFETY: The length check guarantees eight readable values.
        let mut acc = unsafe { _mm256_loadu_ps(values.as_ptr()) };
        index += 8;

        while index < body_len {
            // SAFETY: `index < body_len` guarantees eight readable values.
            acc = unsafe { _mm256_min_ps(acc, _mm256_loadu_ps(values.as_ptr().add(index))) };
            index += 8;
        }

        let mut lanes = [0.0_f32; 8];
        // SAFETY: `lanes` holds exactly eight f32 values.
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
        let mut minimum = lanes.into_iter().fold(f32::INFINITY, f32::min);

        while index < values.len() {
            minimum = minimum.min(values[index]);
            index += 1;
        }

        minimum
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn max_f32(values: &[f32]) -> f32 {
        if values.len() < 8 {
            return values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        }

        let body_len = values.len() / 8 * 8;
        let mut index = 0;
        // SAFETY: The length check guarantees eight readable values.
        let mut acc = unsafe { _mm256_loadu_ps(values.as_ptr()) };
        index += 8;

        while index < body_len {
            // SAFETY: `index < body_len` guarantees eight readable values.
            acc = unsafe { _mm256_max_ps(acc, _mm256_loadu_ps(values.as_ptr().add(index))) };
            index += 8;
        }

        let mut lanes = [0.0_f32; 8];
        // SAFETY: `lanes` holds exactly eight f32 values.
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
        let mut maximum = lanes.into_iter().fold(f32::NEG_INFINITY, f32::max);

        while index < values.len() {
            maximum = maximum.max(values[index]);
            index += 1;
        }

        maximum
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn prod_f64(values: &[f64]) -> f64 {
        if values.len() < 4 {
            return values.iter().copied().product();
        }

        let body_len = values.len() / 4 * 4;
        // SAFETY: This function requires AVX support.
        let mut acc = { _mm256_set1_pd(1.0) };
        let mut index = 0;

        while index < body_len {
            // SAFETY: `index < body_len` guarantees four readable values.
            acc = unsafe { _mm256_mul_pd(acc, _mm256_loadu_pd(values.as_ptr().add(index))) };
            index += 4;
        }

        let mut lanes = [1.0_f64; 4];
        // SAFETY: `lanes` holds exactly four f64 values.
        unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), acc) };
        let mut total: f64 = lanes.into_iter().product();

        while index < values.len() {
            total *= values[index];
            index += 1;
        }

        total
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn min_f64(values: &[f64]) -> f64 {
        if values.len() < 4 {
            return values.iter().copied().fold(f64::INFINITY, f64::min);
        }

        let body_len = values.len() / 4 * 4;
        let mut index = 0;
        // SAFETY: The length check guarantees four readable values.
        let mut acc = unsafe { _mm256_loadu_pd(values.as_ptr()) };
        index += 4;

        while index < body_len {
            // SAFETY: `index < body_len` guarantees four readable values.
            acc = unsafe { _mm256_min_pd(acc, _mm256_loadu_pd(values.as_ptr().add(index))) };
            index += 4;
        }

        let mut lanes = [0.0_f64; 4];
        // SAFETY: `lanes` holds exactly four f64 values.
        unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), acc) };
        let mut minimum = lanes.into_iter().fold(f64::INFINITY, f64::min);

        while index < values.len() {
            minimum = minimum.min(values[index]);
            index += 1;
        }

        minimum
    }

    #[target_feature(enable = "avx")]
    pub(super) unsafe fn max_f64(values: &[f64]) -> f64 {
        if values.len() < 4 {
            return values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        }

        let body_len = values.len() / 4 * 4;
        let mut index = 0;
        // SAFETY: The length check guarantees four readable values.
        let mut acc = unsafe { _mm256_loadu_pd(values.as_ptr()) };
        index += 4;

        while index < body_len {
            // SAFETY: `index < body_len` guarantees four readable values.
            acc = unsafe { _mm256_max_pd(acc, _mm256_loadu_pd(values.as_ptr().add(index))) };
            index += 4;
        }

        let mut lanes = [0.0_f64; 4];
        // SAFETY: `lanes` holds exactly four f64 values.
        unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), acc) };
        let mut maximum = lanes.into_iter().fold(f64::NEG_INFINITY, f64::max);

        while index < values.len() {
            maximum = maximum.max(values[index]);
            index += 1;
        }

        maximum
    }

    #[target_feature(enable = "avx")]
    unsafe fn map_f32(
        lhs: &[f32],
        rhs: &[f32],
        out: &mut [f32],
        op: unsafe fn(__m256, __m256) -> __m256,
    ) {
        let body_len = lhs.len() / 8 * 8;
        let mut index = 0;

        while index < body_len {
            // SAFETY: `index < body_len` guarantees eight readable inputs and writable outputs.
            unsafe {
                let left = _mm256_loadu_ps(lhs.as_ptr().add(index));
                let right = _mm256_loadu_ps(rhs.as_ptr().add(index));
                _mm256_storeu_ps(out.as_mut_ptr().add(index), op(left, right));
            }
            index += 8;
        }

        while index < lhs.len() {
            out[index] = op_scalar_f32(lhs[index], rhs[index], op);
            index += 1;
        }
    }

    #[target_feature(enable = "avx")]
    unsafe fn map_scalar_f32(
        input: &[f32],
        scalar: f32,
        out: &mut [f32],
        op: unsafe fn(__m256, __m256) -> __m256,
    ) {
        let body_len = input.len() / 8 * 8;
        // SAFETY: This function requires AVX support.
        let scalar_vector = { _mm256_set1_ps(scalar) };
        let mut index = 0;

        while index < body_len {
            // SAFETY: `index < body_len` guarantees eight readable inputs and writable outputs.
            unsafe {
                let values = _mm256_loadu_ps(input.as_ptr().add(index));
                _mm256_storeu_ps(out.as_mut_ptr().add(index), op(values, scalar_vector));
            }
            index += 8;
        }

        while index < input.len() {
            out[index] = op_scalar_f32(input[index], scalar, op);
            index += 1;
        }
    }

    #[target_feature(enable = "avx")]
    unsafe fn map_f64(
        lhs: &[f64],
        rhs: &[f64],
        out: &mut [f64],
        op: unsafe fn(__m256d, __m256d) -> __m256d,
    ) {
        let body_len = lhs.len() / 4 * 4;
        let mut index = 0;

        while index < body_len {
            // SAFETY: `index < body_len` guarantees four readable inputs and writable outputs.
            unsafe {
                let left = _mm256_loadu_pd(lhs.as_ptr().add(index));
                let right = _mm256_loadu_pd(rhs.as_ptr().add(index));
                _mm256_storeu_pd(out.as_mut_ptr().add(index), op(left, right));
            }
            index += 4;
        }

        while index < lhs.len() {
            out[index] = op_scalar_f64(lhs[index], rhs[index], op);
            index += 1;
        }
    }

    #[target_feature(enable = "avx")]
    unsafe fn map_scalar_f64(
        input: &[f64],
        scalar: f64,
        out: &mut [f64],
        op: unsafe fn(__m256d, __m256d) -> __m256d,
    ) {
        let body_len = input.len() / 4 * 4;
        // SAFETY: This function requires AVX support.
        let scalar_vector = { _mm256_set1_pd(scalar) };
        let mut index = 0;

        while index < body_len {
            // SAFETY: `index < body_len` guarantees four readable inputs and writable outputs.
            unsafe {
                let values = _mm256_loadu_pd(input.as_ptr().add(index));
                _mm256_storeu_pd(out.as_mut_ptr().add(index), op(values, scalar_vector));
            }
            index += 4;
        }

        while index < input.len() {
            out[index] = op_scalar_f64(input[index], scalar, op);
            index += 1;
        }
    }

    #[inline]
    fn op_scalar_f32(lhs: f32, rhs: f32, op: unsafe fn(__m256, __m256) -> __m256) -> f32 {
        if std::ptr::fn_addr_eq(op, _mm256_add_ps as unsafe fn(__m256, __m256) -> __m256) {
            lhs + rhs
        } else {
            lhs * rhs
        }
    }

    #[inline]
    fn op_scalar_f64(lhs: f64, rhs: f64, op: unsafe fn(__m256d, __m256d) -> __m256d) -> f64 {
        if std::ptr::fn_addr_eq(op, _mm256_add_pd as unsafe fn(__m256d, __m256d) -> __m256d) {
            lhs + rhs
        } else {
            lhs * rhs
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        add_contiguous, add_scalar_contiguous, max_contiguous, mean_contiguous, min_contiguous,
        mul_contiguous, mul_scalar_contiguous, prod_contiguous, sum_contiguous,
    };

    macro_rules! assert_floating_simd_matches_scalar {
        ($name:ident, $ty:ty, $tolerance:expr) => {
            #[test]
            fn $name() {
                const LEN: usize = 259;
                let lhs: Vec<$ty> = (0..LEN).map(|index| index as $ty * 0.25 - 31.5).collect();
                let rhs: Vec<$ty> = (0..LEN).map(|index| index as $ty * -0.125 + 15.25).collect();
                let mut add_out = vec![0.0; LEN];
                let mut mul_out = vec![0.0; LEN];

                add_contiguous(&lhs, &rhs, &mut add_out);
                mul_contiguous(&lhs, &rhs, &mut mul_out);

                let expected_add: Vec<$ty> =
                    lhs.iter().zip(&rhs).map(|(&left, &right)| left + right).collect();
                let expected_mul: Vec<$ty> =
                    lhs.iter().zip(&rhs).map(|(&left, &right)| left * right).collect();
                assert_eq!(add_out, expected_add);
                assert_eq!(mul_out, expected_mul);

                let product_values: Vec<$ty> =
                    (0..LEN).map(|index| 1.0 + (index % 3) as $ty * 0.001).collect();
                let expected_sum = lhs.iter().copied().fold(0.0, |total, value| total + value);
                let expected_product =
                    product_values.iter().copied().fold(1.0, |total, value| total * value);
                let expected_min = lhs.iter().copied().fold(lhs[0], |min, value| min.min(value));
                let expected_max = lhs.iter().copied().fold(lhs[0], |max, value| max.max(value));

                assert!((sum_contiguous(&lhs) - expected_sum).abs() <= $tolerance);
                assert!((prod_contiguous(&product_values) - expected_product).abs() <= $tolerance);
                assert_eq!(min_contiguous(&lhs, "min").unwrap(), expected_min);
                assert_eq!(max_contiguous(&lhs, "max").unwrap(), expected_max);

                let mut nan_values = lhs;
                nan_values[LEN - 2] = <$ty>::NAN;
                add_contiguous(&nan_values, &rhs, &mut add_out);
                mul_contiguous(&nan_values, &rhs, &mut mul_out);

                assert!(add_out[LEN - 2].is_nan());
                assert!(mul_out[LEN - 2].is_nan());
                assert!(sum_contiguous(&nan_values).is_nan());
                assert!(prod_contiguous(&nan_values).is_nan());
                assert!(min_contiguous(&nan_values, "min").unwrap().is_nan());
                assert!(max_contiguous(&nan_values, "max").unwrap().is_nan());
            }
        };
    }

    assert_floating_simd_matches_scalar!(f32_simd_matches_scalar_with_tails_and_nans, f32, 1e-4);
    assert_floating_simd_matches_scalar!(f64_simd_matches_scalar_with_tails_and_nans, f64, 1e-12);

    #[test]
    fn f32_simd_candidates_match_expected_results() {
        let lhs = vec![1.0_f32; 19];
        let rhs = vec![2.0_f32; 19];
        let mut add_out = vec![0.0_f32; 19];
        let mut mul_out = vec![0.0_f32; 19];

        add_contiguous(&lhs, &rhs, &mut add_out);
        mul_contiguous(&lhs, &rhs, &mut mul_out);

        assert!(add_out.iter().all(|&value| value == 3.0));
        assert!(mul_out.iter().all(|&value| value == 2.0));
        assert_eq!(sum_contiguous(&rhs), 38.0);
        assert_eq!(prod_contiguous(&[2.0_f32; 4]), 16.0);
        assert_eq!(min_contiguous(&rhs, "min").unwrap(), 2.0);
        assert_eq!(max_contiguous(&rhs, "max").unwrap(), 2.0);
        assert_eq!(mean_contiguous(&rhs, "mean").unwrap(), 2.0);
    }

    #[test]
    fn scalar_paths_cover_integer_fallbacks() {
        let lhs = vec![1_i32, 2, 3, 4, 5];
        let rhs = vec![2_i32, 3, 4, 5, 6];
        let mut add_out = vec![0_i32; 5];
        let mut mul_out = vec![0_i32; 5];

        add_contiguous(&lhs, &rhs, &mut add_out);
        mul_contiguous(&lhs, &rhs, &mut mul_out);
        add_scalar_contiguous(&lhs, 2, &mut add_out);
        mul_scalar_contiguous(&lhs, 2, &mut mul_out);

        assert_eq!(sum_contiguous(&lhs), 15);
        assert_eq!(prod_contiguous(&[1_i32, 2, 3, 4]), 24);
        assert_eq!(add_out, vec![3, 4, 5, 6, 7]);
        assert_eq!(mul_out, vec![2, 4, 6, 8, 10]);
    }
}
