use num_traits::ToPrimitive;

use crate::{AtlasNdError, AtlasNdResult, Numeric};

const CONTIGUOUS_LANES: usize = 8;

pub(crate) fn map_binary_contiguous<T, F>(lhs: &[T], rhs: &[T], out: &mut [T], op: F)
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
{
    debug_assert_eq!(lhs.len(), rhs.len());
    debug_assert_eq!(lhs.len(), out.len());

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
        index += CONTIGUOUS_LANES;
    }

    while index < len {
        out[index] = op(lhs[index], rhs[index]);
        index += 1;
    }
}

pub(crate) fn map_scalar_contiguous<T, F>(input: &[T], scalar: T, out: &mut [T], op: F)
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
{
    debug_assert_eq!(input.len(), out.len());

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
        index += CONTIGUOUS_LANES;
    }

    while index < len {
        out[index] = op(input[index], scalar);
        index += 1;
    }
}

pub(crate) fn sum_contiguous<T: Numeric>(values: &[T]) -> T {
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
        acc0 += values[index];
        acc1 += values[index + 1];
        acc2 += values[index + 2];
        acc3 += values[index + 3];
        acc4 += values[index + 4];
        acc5 += values[index + 5];
        acc6 += values[index + 6];
        acc7 += values[index + 7];
        index += CONTIGUOUS_LANES;
    }

    let mut total = acc0 + acc1;
    total += acc2 + acc3;
    total += acc4 + acc5;
    total += acc6 + acc7;

    while index < len {
        total += values[index];
        index += 1;
    }

    total
}

pub(crate) fn prod_contiguous<T: Numeric>(values: &[T]) -> T {
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
        acc0 *= values[index];
        acc1 *= values[index + 1];
        acc2 *= values[index + 2];
        acc3 *= values[index + 3];
        acc4 *= values[index + 4];
        acc5 *= values[index + 5];
        acc6 *= values[index + 6];
        acc7 *= values[index + 7];
        index += CONTIGUOUS_LANES;
    }

    let mut total = acc0 * acc1;
    total *= acc2 * acc3;
    total *= acc4 * acc5;
    total *= acc6 * acc7;

    while index < len {
        total *= values[index];
        index += 1;
    }

    total
}

pub(crate) fn min_contiguous<T>(values: &[T], op: &'static str) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    let mut iter = values.iter().copied();
    let mut minimum = iter.next().ok_or(AtlasNdError::EmptyReduction { op })?;

    for value in iter {
        if value < minimum {
            minimum = value;
        }
    }

    Ok(minimum)
}

pub(crate) fn max_contiguous<T>(values: &[T], op: &'static str) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    let mut iter = values.iter().copied();
    let mut maximum = iter.next().ok_or(AtlasNdError::EmptyReduction { op })?;

    for value in iter {
        if value > maximum {
            maximum = value;
        }
    }

    Ok(maximum)
}

pub(crate) fn mean_contiguous<T>(values: &[T], op: &'static str) -> AtlasNdResult<f64>
where
    T: Numeric + ToPrimitive,
{
    if values.is_empty() {
        return Err(AtlasNdError::EmptyReduction { op });
    }

    let len = values.len();
    let body_len = body_len(len);
    let mut acc0 = 0.0_f64;
    let mut acc1 = 0.0_f64;
    let mut acc2 = 0.0_f64;
    let mut acc3 = 0.0_f64;
    let mut acc4 = 0.0_f64;
    let mut acc5 = 0.0_f64;
    let mut acc6 = 0.0_f64;
    let mut acc7 = 0.0_f64;
    let mut index = 0;

    while index < body_len {
        acc0 += values[index].to_f64().ok_or(AtlasNdError::NumericConversionFailed { op })?;
        acc1 += values[index + 1].to_f64().ok_or(AtlasNdError::NumericConversionFailed { op })?;
        acc2 += values[index + 2].to_f64().ok_or(AtlasNdError::NumericConversionFailed { op })?;
        acc3 += values[index + 3].to_f64().ok_or(AtlasNdError::NumericConversionFailed { op })?;
        acc4 += values[index + 4].to_f64().ok_or(AtlasNdError::NumericConversionFailed { op })?;
        acc5 += values[index + 5].to_f64().ok_or(AtlasNdError::NumericConversionFailed { op })?;
        acc6 += values[index + 6].to_f64().ok_or(AtlasNdError::NumericConversionFailed { op })?;
        acc7 += values[index + 7].to_f64().ok_or(AtlasNdError::NumericConversionFailed { op })?;
        index += CONTIGUOUS_LANES;
    }

    let mut total = acc0 + acc1;
    total += acc2 + acc3;
    total += acc4 + acc5;
    total += acc6 + acc7;

    while index < len {
        total += values[index].to_f64().ok_or(AtlasNdError::NumericConversionFailed { op })?;
        index += 1;
    }

    Ok(total / len as f64)
}

#[inline]
fn body_len(len: usize) -> usize {
    len / CONTIGUOUS_LANES * CONTIGUOUS_LANES
}
