use num_traits::ToPrimitive;
use rayon::prelude::*;

use crate::{
    AtlasNdError, AtlasNdResult, Numeric,
    internal::simd,
    internal::{for_each_value, layout::dense_storage_slice},
    layout::element_count,
};

use super::dispatch::{
    ensure_non_empty_reduction, parallel_reduction_chunk_len, should_parallelize_reduction,
};

pub(super) fn sum_contiguous<T: Numeric>(values: &[T]) -> T {
    if should_parallelize_reduction(values.len()) {
        let partials: Vec<T> =
            values.par_chunks(parallel_reduction_chunk_len()).map(simd::sum_contiguous).collect();

        return partials.into_iter().fold(T::zero(), |total, partial| total + partial);
    }

    simd::sum_contiguous(values)
}

pub(super) fn prod_contiguous<T: Numeric>(values: &[T]) -> T {
    if should_parallelize_reduction(values.len()) {
        let partials: Vec<T> =
            values.par_chunks(parallel_reduction_chunk_len()).map(simd::prod_contiguous).collect();

        return partials.into_iter().fold(T::one(), |total, partial| total * partial);
    }

    simd::prod_contiguous(values)
}

pub(super) fn min_contiguous<T>(values: &[T], op: &'static str) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    if should_parallelize_reduction(values.len()) {
        let partials: Vec<T> = values
            .par_chunks(parallel_reduction_chunk_len())
            .map(|chunk| {
                simd::min_contiguous(chunk, op).expect("parallel reduction chunks are non-empty")
            })
            .collect();

        return fold_min(partials.into_iter(), op);
    }

    simd::min_contiguous(values, op)
}

pub(super) fn max_contiguous<T>(values: &[T], op: &'static str) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    if should_parallelize_reduction(values.len()) {
        let partials: Vec<T> = values
            .par_chunks(parallel_reduction_chunk_len())
            .map(|chunk| {
                simd::max_contiguous(chunk, op).expect("parallel reduction chunks are non-empty")
            })
            .collect();

        return fold_max(partials.into_iter(), op);
    }

    simd::max_contiguous(values, op)
}

pub(super) fn sum_all<T: Numeric>(
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> T {
    if element_count(shape) == 0 {
        return T::zero();
    }

    if let Some(values) = dense_storage_slice(data, offset, shape, strides) {
        return sum_contiguous(values);
    }

    let mut total = T::zero();
    for_each_value(data, offset, shape, strides, |value| {
        total += *value;
    });
    total
}

pub(super) fn prod_all<T: Numeric>(
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> T {
    if element_count(shape) == 0 {
        return T::one();
    }

    if let Some(values) = dense_storage_slice(data, offset, shape, strides) {
        return prod_contiguous(values);
    }

    let mut total = T::one();
    for_each_value(data, offset, shape, strides, |value| {
        total *= *value;
    });
    total
}

pub(super) fn min_all<T>(
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    ensure_non_empty_reduction(element_count(shape), "min")?;

    if let Some(values) = dense_storage_slice(data, offset, shape, strides) {
        return min_contiguous(values, "min");
    }

    let mut minimum = None;

    for_each_value(data, offset, shape, strides, |value| {
        minimum = Some(match minimum {
            Some(current) if current < *value => current,
            Some(_) | None => *value,
        });
    });

    minimum.ok_or(AtlasNdError::EmptyReduction { op: "min" })
}

pub(super) fn max_all<T>(
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    ensure_non_empty_reduction(element_count(shape), "max")?;

    if let Some(values) = dense_storage_slice(data, offset, shape, strides) {
        return max_contiguous(values, "max");
    }

    let mut maximum = None;

    for_each_value(data, offset, shape, strides, |value| {
        maximum = Some(match maximum {
            Some(current) if current > *value => current,
            Some(_) | None => *value,
        });
    });

    maximum.ok_or(AtlasNdError::EmptyReduction { op: "max" })
}

pub(super) fn mean_all<T>(
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> AtlasNdResult<f64>
where
    T: Numeric + ToPrimitive,
{
    super::mean::mean_all(data, offset, shape, strides)
}

fn fold_min<T>(values: impl IntoIterator<Item = T>, op: &'static str) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    let mut iter = values.into_iter();
    let mut current = iter.next().ok_or(AtlasNdError::EmptyReduction { op })?;

    for value in iter {
        if value < current {
            current = value;
        }
    }

    Ok(current)
}

fn fold_max<T>(values: impl IntoIterator<Item = T>, op: &'static str) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    let mut iter = values.into_iter();
    let mut current = iter.next().ok_or(AtlasNdError::EmptyReduction { op })?;

    for value in iter {
        if value > current {
            current = value;
        }
    }

    Ok(current)
}
