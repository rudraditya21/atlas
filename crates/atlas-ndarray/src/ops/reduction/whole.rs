use num_traits::ToPrimitive;
use rayon::prelude::*;

use super::{
    dispatch::{
        parallel_reduction_chunk_len, should_parallelize_reduction, with_reduction_scratch_f32,
        with_reduction_scratch_f64,
    },
    metadata::WholeReductionMetadata,
};
use crate::{
    AtlasNdError, AtlasNdResult, ElementwiseArithmetic, Numeric,
    internal::{layout::dense_storage_slice, logical_span_iter, simd},
};

pub(super) fn sum_contiguous<T: ElementwiseArithmetic>(values: &[T]) -> T {
    if should_parallelize_reduction(values.len()) {
        if simd::is_f32::<T>() {
            return simd::cast_value_exact(parallel_sum_f32(simd::cast_slice(values)));
        }

        if simd::is_f64::<T>() {
            return simd::cast_value_exact(parallel_sum_f64(simd::cast_slice(values)));
        }

        return values
            .par_chunks(parallel_reduction_chunk_len())
            .map(simd::sum_contiguous)
            .reduce(T::zero, ElementwiseArithmetic::elementwise_add);
    }

    simd::sum_contiguous(values)
}

pub(super) fn prod_contiguous<T: ElementwiseArithmetic>(values: &[T]) -> T {
    if should_parallelize_reduction(values.len()) {
        if simd::is_f32::<T>() {
            return simd::cast_value_exact(parallel_prod_f32(simd::cast_slice(values)));
        }

        if simd::is_f64::<T>() {
            return simd::cast_value_exact(parallel_prod_f64(simd::cast_slice(values)));
        }

        return values
            .par_chunks(parallel_reduction_chunk_len())
            .map(simd::prod_contiguous)
            .reduce(T::one, ElementwiseArithmetic::elementwise_mul);
    }

    simd::prod_contiguous(values)
}

pub(super) fn min_contiguous<T>(values: &[T], op: &'static str) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    if should_parallelize_reduction(values.len()) {
        if simd::is_f32::<T>() {
            return Ok(simd::cast_value_exact(parallel_extreme_f32(
                simd::cast_slice(values),
                op,
                simd::min_contiguous,
                simd::min_propagating,
            )));
        }

        if simd::is_f64::<T>() {
            return Ok(simd::cast_value_exact(parallel_extreme_f64(
                simd::cast_slice(values),
                op,
                simd::min_contiguous,
                simd::min_propagating,
            )));
        }

        return values
            .par_chunks(parallel_reduction_chunk_len())
            .map(|chunk| {
                simd::min_contiguous(chunk, op).expect("parallel reduction chunks are non-empty")
            })
            .reduce_with(simd::min_propagating)
            .ok_or(AtlasNdError::EmptyReduction { op });
    }

    simd::min_contiguous(values, op)
}

pub(super) fn max_contiguous<T>(values: &[T], op: &'static str) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    if should_parallelize_reduction(values.len()) {
        if simd::is_f32::<T>() {
            return Ok(simd::cast_value_exact(parallel_extreme_f32(
                simd::cast_slice(values),
                op,
                simd::max_contiguous,
                simd::max_propagating,
            )));
        }

        if simd::is_f64::<T>() {
            return Ok(simd::cast_value_exact(parallel_extreme_f64(
                simd::cast_slice(values),
                op,
                simd::max_contiguous,
                simd::max_propagating,
            )));
        }

        return values
            .par_chunks(parallel_reduction_chunk_len())
            .map(|chunk| {
                simd::max_contiguous(chunk, op).expect("parallel reduction chunks are non-empty")
            })
            .reduce_with(simd::max_propagating)
            .ok_or(AtlasNdError::EmptyReduction { op });
    }

    simd::max_contiguous(values, op)
}

pub(super) fn sum_all<T: ElementwiseArithmetic>(
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

    let total = logical_span_iter(data, offset, shape, strides)
        .fold(T::zero(), |total, span| total.elementwise_add(sum_contiguous(span)));
    Ok(total)
}

pub(super) fn prod_all<T: ElementwiseArithmetic>(
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

    let total = logical_span_iter(data, offset, shape, strides)
        .fold(T::one(), |total, span| total.elementwise_mul(prod_contiguous(span)));
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

    for span in logical_span_iter(data, offset, shape, strides) {
        let value = min_contiguous(span, "min")?;
        minimum = Some(minimum.map_or(value, |current| simd::min_propagating(current, value)));
    }

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

    for span in logical_span_iter(data, offset, shape, strides) {
        let value = max_contiguous(span, "max")?;
        maximum = Some(maximum.map_or(value, |current| simd::max_propagating(current, value)));
    }

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

fn parallel_sum_f32(values: &[f32]) -> f32 {
    let chunk_len = parallel_reduction_chunk_len();
    let partial_count = values.len().div_ceil(chunk_len);
    with_reduction_scratch_f32(partial_count, |partials| {
        partials
            .par_iter_mut()
            .zip(values.par_chunks(chunk_len))
            .for_each(|(partial, chunk)| *partial = simd::sum_contiguous(chunk));

        let mut total = simd::CompensatedSum::new();
        for &partial in partials.iter() {
            total.add(f64::from(partial));
        }
        total.finish() as f32
    })
}

fn parallel_sum_f64(values: &[f64]) -> f64 {
    let chunk_len = parallel_reduction_chunk_len();
    let partial_count = values.len().div_ceil(chunk_len);
    with_reduction_scratch_f64(partial_count, |partials| {
        partials
            .par_iter_mut()
            .zip(values.par_chunks(chunk_len))
            .for_each(|(partial, chunk)| *partial = simd::sum_contiguous(chunk));

        let mut total = simd::CompensatedSum::new();
        for &partial in partials.iter() {
            total.add(partial);
        }
        total.finish()
    })
}

fn parallel_prod_f32(values: &[f32]) -> f32 {
    let chunk_len = parallel_reduction_chunk_len();
    let partial_count = values.len().div_ceil(chunk_len);
    with_reduction_scratch_f32(partial_count, |partials| {
        partials
            .par_iter_mut()
            .zip(values.par_chunks(chunk_len))
            .for_each(|(partial, chunk)| *partial = simd::prod_contiguous(chunk));
        partials.iter().copied().product()
    })
}

fn parallel_prod_f64(values: &[f64]) -> f64 {
    let chunk_len = parallel_reduction_chunk_len();
    let partial_count = values.len().div_ceil(chunk_len);
    with_reduction_scratch_f64(partial_count, |partials| {
        partials
            .par_iter_mut()
            .zip(values.par_chunks(chunk_len))
            .for_each(|(partial, chunk)| *partial = simd::prod_contiguous(chunk));
        partials.iter().copied().product()
    })
}

fn parallel_extreme_f32(
    values: &[f32],
    op: &'static str,
    reduce_chunk: fn(&[f32], &'static str) -> AtlasNdResult<f32>,
    combine: fn(f32, f32) -> f32,
) -> f32 {
    let chunk_len = parallel_reduction_chunk_len();
    let partial_count = values.len().div_ceil(chunk_len);
    with_reduction_scratch_f32(partial_count, |partials| {
        partials.par_iter_mut().zip(values.par_chunks(chunk_len)).for_each(|(partial, chunk)| {
            *partial = reduce_chunk(chunk, op).expect("parallel reduction chunks are non-empty");
        });

        partials.iter().copied().reduce(combine).expect("parallel reductions have partials")
    })
}

fn parallel_extreme_f64(
    values: &[f64],
    op: &'static str,
    reduce_chunk: fn(&[f64], &'static str) -> AtlasNdResult<f64>,
    combine: fn(f64, f64) -> f64,
) -> f64 {
    let chunk_len = parallel_reduction_chunk_len();
    let partial_count = values.len().div_ceil(chunk_len);
    with_reduction_scratch_f64(partial_count, |partials| {
        partials.par_iter_mut().zip(values.par_chunks(chunk_len)).for_each(|(partial, chunk)| {
            *partial = reduce_chunk(chunk, op).expect("parallel reduction chunks are non-empty");
        });

        partials.iter().copied().reduce(combine).expect("parallel reductions have partials")
    })
}

fn sum_all_f32(data: &[f32], offset: usize, shape: &[usize], strides: &[usize]) -> f32 {
    let mut total = simd::CompensatedSum::new();

    for span in logical_span_iter(data, offset, shape, strides) {
        total.add(f64::from(sum_contiguous(span)));
    }

    total.finish() as f32
}

fn sum_all_f64(data: &[f64], offset: usize, shape: &[usize], strides: &[usize]) -> f64 {
    let mut total = simd::CompensatedSum::new();

    for span in logical_span_iter(data, offset, shape, strides) {
        total.add(sum_contiguous(span));
    }

    total.finish()
}
