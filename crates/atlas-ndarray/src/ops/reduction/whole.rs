use num_traits::ToPrimitive;
use rayon::prelude::*;

use crate::{
    AtlasNdError, AtlasNdResult, Numeric,
    internal::simd,
    internal::{for_each_value, layout::dense_storage_slice},
};

use super::dispatch::{parallel_reduction_chunk_len, should_parallelize_reduction};
use super::metadata::WholeReductionMetadata;

pub(super) fn sum_contiguous<T: Numeric>(values: &[T]) -> T {
    if should_parallelize_reduction(values.len()) {
        let partials: Vec<T> =
            values.par_chunks(parallel_reduction_chunk_len()).map(simd::sum_contiguous).collect();

        if simd::is_f32::<T>() {
            let mut total = simd::CompensatedSum::new();
            for value in simd::cast_slice::<T, f32>(&partials) {
                total.add(f64::from(*value));
            }
            return simd::cast_value_exact(total.finish() as f32);
        }

        if simd::is_f64::<T>() {
            let mut total = simd::CompensatedSum::new();
            for value in simd::cast_slice::<T, f64>(&partials) {
                total.add(*value);
            }
            return simd::cast_value_exact(total.finish());
        }

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

        return fold_min(partials, op);
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

        return fold_max(partials, op);
    }

    simd::max_contiguous(values, op)
}

pub(super) fn sum_all<T: Numeric>(
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> AtlasNdResult<T> {
    let metadata = WholeReductionMetadata::from_shape(shape);
    metadata.require_non_empty("sum")?;

    if let Some(values) = dense_storage_slice(data, offset, shape, strides) {
        return Ok(sum_contiguous(values));
    }

    if simd::is_f32::<T>() {
        return Ok(simd::cast_value_exact(sum_all_f32(
            simd::cast_slice(data),
            offset,
            shape,
            strides,
        )));
    }

    if simd::is_f64::<T>() {
        return Ok(simd::cast_value_exact(sum_all_f64(
            simd::cast_slice(data),
            offset,
            shape,
            strides,
        )));
    }

    let mut total = T::zero();
    for_each_value(data, offset, shape, strides, |value| {
        total += *value;
    });
    Ok(total)
}

pub(super) fn prod_all<T: Numeric>(
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> AtlasNdResult<T> {
    let metadata = WholeReductionMetadata::from_shape(shape);
    metadata.require_non_empty("prod")?;

    if let Some(values) = dense_storage_slice(data, offset, shape, strides) {
        return Ok(prod_contiguous(values));
    }

    let mut total = T::one();
    for_each_value(data, offset, shape, strides, |value| {
        total *= *value;
    });
    Ok(total)
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
    let metadata = WholeReductionMetadata::from_shape(shape);
    metadata.require_non_empty("min")?;

    if let Some(values) = dense_storage_slice(data, offset, shape, strides) {
        return min_contiguous(values, "min");
    }

    let mut minimum = None;

    for_each_value(data, offset, shape, strides, |value| {
        minimum = Some(minimum.map_or(*value, |current| simd::min_propagating(current, *value)));
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
    let metadata = WholeReductionMetadata::from_shape(shape);
    metadata.require_non_empty("max")?;

    if let Some(values) = dense_storage_slice(data, offset, shape, strides) {
        return max_contiguous(values, "max");
    }

    let mut maximum = None;

    for_each_value(data, offset, shape, strides, |value| {
        maximum = Some(maximum.map_or(*value, |current| simd::max_propagating(current, *value)));
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
        current = simd::min_propagating(current, value);
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
        current = simd::max_propagating(current, value);
    }

    Ok(current)
}

fn sum_all_f32(data: &[f32], offset: usize, shape: &[usize], strides: &[usize]) -> f32 {
    let mut total = simd::CompensatedSum::new();

    for_each_value(data, offset, shape, strides, |value| {
        total.add(f64::from(*value));
    });

    total.finish() as f32
}

fn sum_all_f64(data: &[f64], offset: usize, shape: &[usize], strides: &[usize]) -> f64 {
    let mut total = simd::CompensatedSum::new();

    for_each_value(data, offset, shape, strides, |value| {
        total.add(*value);
    });

    total.finish()
}
