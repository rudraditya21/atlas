use crate::internal::simd;
use num_traits::ToPrimitive;
use rayon::prelude::*;

use crate::{
    AtlasNdResult, AxisIndex, ElementwiseArithmetic, NDArray, Numeric,
    internal::{layout::LayoutKind, offset_iter},
};

use super::{
    dispatch::should_parallelize_reduction,
    mean::{mean_axis_contiguous, mean_axis_dense_contiguous, mean_axis_strided},
    metadata::AxisReductionMetadata,
    whole::{max_contiguous, min_contiguous, prod_contiguous, sum_contiguous},
};

pub(super) fn sum_axis_impl<T: ElementwiseArithmetic>(
    data: &[T],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<T>> {
    let metadata = AxisReductionMetadata::new(shape, strides, axis, false)?;
    dispatch_sum_axis(data, base_offset, metadata)
}

pub(super) fn sum_axis_keepdims_impl<T: ElementwiseArithmetic>(
    data: &[T],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<T>> {
    let metadata = AxisReductionMetadata::new(shape, strides, axis, true)?;
    dispatch_sum_axis(data, base_offset, metadata)
}

fn dispatch_sum_axis<T: ElementwiseArithmetic>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>> {
    metadata.require_non_empty("sum")?;

    match (metadata.source_layout, metadata.axis_layout) {
        (LayoutKind::Contiguous, _) => sum_axis_dense_contiguous(data, base_offset, metadata),
        (LayoutKind::Strided, LayoutKind::Contiguous) => {
            sum_axis_contiguous(data, base_offset, metadata)
        }
        (LayoutKind::Strided, LayoutKind::Strided) => sum_axis_strided(data, base_offset, metadata),
    }
}

pub(super) fn prod_axis_impl<T: ElementwiseArithmetic>(
    data: &[T],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<T>> {
    let metadata = AxisReductionMetadata::new(shape, strides, axis, false)?;
    dispatch_prod_axis(data, base_offset, metadata)
}

pub(super) fn prod_axis_keepdims_impl<T: ElementwiseArithmetic>(
    data: &[T],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<T>> {
    let metadata = AxisReductionMetadata::new(shape, strides, axis, true)?;
    dispatch_prod_axis(data, base_offset, metadata)
}

fn dispatch_prod_axis<T: ElementwiseArithmetic>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>> {
    metadata.require_non_empty("prod")?;

    match (metadata.source_layout, metadata.axis_layout) {
        (LayoutKind::Contiguous, _) => prod_axis_dense_contiguous(data, base_offset, metadata),
        (LayoutKind::Strided, LayoutKind::Contiguous) => {
            prod_axis_contiguous(data, base_offset, metadata)
        }
        (LayoutKind::Strided, LayoutKind::Strided) => {
            prod_axis_strided(data, base_offset, metadata)
        }
    }
}

pub(super) fn min_axis_impl<T>(
    data: &[T],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
{
    let metadata = AxisReductionMetadata::new(shape, strides, axis, false)?;
    dispatch_min_axis(data, base_offset, metadata)
}

pub(super) fn min_axis_keepdims_impl<T>(
    data: &[T],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
{
    let metadata = AxisReductionMetadata::new(shape, strides, axis, true)?;
    dispatch_min_axis(data, base_offset, metadata)
}

fn dispatch_min_axis<T>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
{
    metadata.require_non_empty("min")?;

    match (metadata.source_layout, metadata.axis_layout) {
        (LayoutKind::Contiguous, _) => min_axis_dense_contiguous(data, base_offset, metadata),
        (LayoutKind::Strided, LayoutKind::Contiguous) => {
            min_axis_contiguous(data, base_offset, metadata)
        }
        (LayoutKind::Strided, LayoutKind::Strided) => min_axis_strided(data, base_offset, metadata),
    }
}

pub(super) fn max_axis_impl<T>(
    data: &[T],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
{
    let metadata = AxisReductionMetadata::new(shape, strides, axis, false)?;
    dispatch_max_axis(data, base_offset, metadata)
}

pub(super) fn max_axis_keepdims_impl<T>(
    data: &[T],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
{
    let metadata = AxisReductionMetadata::new(shape, strides, axis, true)?;
    dispatch_max_axis(data, base_offset, metadata)
}

fn dispatch_max_axis<T>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
{
    metadata.require_non_empty("max")?;

    match (metadata.source_layout, metadata.axis_layout) {
        (LayoutKind::Contiguous, _) => max_axis_dense_contiguous(data, base_offset, metadata),
        (LayoutKind::Strided, LayoutKind::Contiguous) => {
            max_axis_contiguous(data, base_offset, metadata)
        }
        (LayoutKind::Strided, LayoutKind::Strided) => max_axis_strided(data, base_offset, metadata),
    }
}

pub(super) fn mean_axis_impl<T>(
    data: &[T],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
{
    let metadata = AxisReductionMetadata::new(shape, strides, axis, false)?;
    dispatch_mean_axis(data, base_offset, metadata)
}

pub(super) fn mean_axis_keepdims_impl<T>(
    data: &[T],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
{
    let metadata = AxisReductionMetadata::new(shape, strides, axis, true)?;
    dispatch_mean_axis(data, base_offset, metadata)
}

fn dispatch_mean_axis<T>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
{
    metadata.require_non_empty("mean")?;

    match (metadata.source_layout, metadata.axis_layout) {
        (LayoutKind::Contiguous, _) => mean_axis_dense_contiguous(data, base_offset, metadata),
        (LayoutKind::Strided, LayoutKind::Contiguous) => {
            mean_axis_contiguous(data, base_offset, metadata)
        }
        (LayoutKind::Strided, LayoutKind::Strided) => {
            mean_axis_strided(data, base_offset, metadata)
        }
    }
}

fn sum_axis_dense_contiguous<T: ElementwiseArithmetic>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>> {
    let values =
        contiguous_region(data, base_offset, metadata.output.len.saturating_mul(metadata.axis_len));

    if simd::is_f32::<T>() {
        return sum_axis_dense_contiguous_f32::<T>(simd::cast_slice(values), metadata);
    }

    if simd::is_f64::<T>() {
        return sum_axis_dense_contiguous_f64::<T>(simd::cast_slice(values), metadata);
    }

    let mut reduced = vec![T::zero(); metadata.output.len];

    if should_parallelize_reduction(metadata.output.len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_chunks_mut(metadata.contiguous_inner_len).enumerate().for_each(
            |(outer, output_row)| {
                let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;

                for (inner, slot) in output_row.iter_mut().enumerate() {
                    let mut total = T::zero();
                    let mut offset = block_start + inner;

                    for _ in 0..metadata.axis_len {
                        total = total.elementwise_add(values[offset]);
                        offset += metadata.contiguous_inner_len;
                    }

                    *slot = total;
                }
            },
        );
    } else {
        for outer in 0..metadata.contiguous_outer_len {
            let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;
            let output_start = outer * metadata.contiguous_inner_len;

            for inner in 0..metadata.contiguous_inner_len {
                let mut total = T::zero();
                let mut offset = block_start + inner;

                for _ in 0..metadata.axis_len {
                    total = total.elementwise_add(values[offset]);
                    offset += metadata.contiguous_inner_len;
                }

                reduced[output_start + inner] = total;
            }
        }
    }

    NDArray::from_shape_vec(metadata.output.shape, reduced)
}

fn prod_axis_dense_contiguous<T: ElementwiseArithmetic>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>> {
    let values =
        contiguous_region(data, base_offset, metadata.output.len.saturating_mul(metadata.axis_len));
    let mut reduced = vec![T::zero(); metadata.output.len];

    if should_parallelize_reduction(metadata.output.len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_chunks_mut(metadata.contiguous_inner_len).enumerate().for_each(
            |(outer, output_row)| {
                let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;

                for (inner, slot) in output_row.iter_mut().enumerate() {
                    let mut total = T::one();
                    let mut offset = block_start + inner;

                    for _ in 0..metadata.axis_len {
                        total = total.elementwise_mul(values[offset]);
                        offset += metadata.contiguous_inner_len;
                    }

                    *slot = total;
                }
            },
        );
    } else {
        for outer in 0..metadata.contiguous_outer_len {
            let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;
            let output_start = outer * metadata.contiguous_inner_len;

            for inner in 0..metadata.contiguous_inner_len {
                let mut total = T::one();
                let mut offset = block_start + inner;

                for _ in 0..metadata.axis_len {
                    total = total.elementwise_mul(values[offset]);
                    offset += metadata.contiguous_inner_len;
                }

                reduced[output_start + inner] = total;
            }
        }
    }

    NDArray::from_shape_vec(metadata.output.shape, reduced)
}

fn min_axis_dense_contiguous<T>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
{
    let values =
        contiguous_region(data, base_offset, metadata.output.len.saturating_mul(metadata.axis_len));
    let mut reduced = vec![T::zero(); metadata.output.len];

    if should_parallelize_reduction(metadata.output.len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_chunks_mut(metadata.contiguous_inner_len).enumerate().for_each(
            |(outer, output_row)| {
                let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;

                for (inner, slot) in output_row.iter_mut().enumerate() {
                    let mut offset = block_start + inner;
                    let mut current = values[offset];
                    offset += metadata.contiguous_inner_len;

                    for _ in 1..metadata.axis_len {
                        let value = values[offset];
                        current = simd::min_propagating(current, value);
                        offset += metadata.contiguous_inner_len;
                    }

                    *slot = current;
                }
            },
        );
    } else {
        for outer in 0..metadata.contiguous_outer_len {
            let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;
            let output_start = outer * metadata.contiguous_inner_len;

            for inner in 0..metadata.contiguous_inner_len {
                let mut offset = block_start + inner;
                let mut current = values[offset];
                offset += metadata.contiguous_inner_len;

                for _ in 1..metadata.axis_len {
                    let value = values[offset];
                    current = simd::min_propagating(current, value);
                    offset += metadata.contiguous_inner_len;
                }

                reduced[output_start + inner] = current;
            }
        }
    }

    NDArray::from_shape_vec(metadata.output.shape, reduced)
}

fn max_axis_dense_contiguous<T>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
{
    let values =
        contiguous_region(data, base_offset, metadata.output.len.saturating_mul(metadata.axis_len));
    let mut reduced = vec![T::zero(); metadata.output.len];

    if should_parallelize_reduction(metadata.output.len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_chunks_mut(metadata.contiguous_inner_len).enumerate().for_each(
            |(outer, output_row)| {
                let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;

                for (inner, slot) in output_row.iter_mut().enumerate() {
                    let mut offset = block_start + inner;
                    let mut current = values[offset];
                    offset += metadata.contiguous_inner_len;

                    for _ in 1..metadata.axis_len {
                        let value = values[offset];
                        current = simd::max_propagating(current, value);
                        offset += metadata.contiguous_inner_len;
                    }

                    *slot = current;
                }
            },
        );
    } else {
        for outer in 0..metadata.contiguous_outer_len {
            let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;
            let output_start = outer * metadata.contiguous_inner_len;

            for inner in 0..metadata.contiguous_inner_len {
                let mut offset = block_start + inner;
                let mut current = values[offset];
                offset += metadata.contiguous_inner_len;

                for _ in 1..metadata.axis_len {
                    let value = values[offset];
                    current = simd::max_propagating(current, value);
                    offset += metadata.contiguous_inner_len;
                }

                reduced[output_start + inner] = current;
            }
        }
    }

    NDArray::from_shape_vec(metadata.output.shape, reduced)
}

fn sum_axis_contiguous<T: ElementwiseArithmetic>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>> {
    let mut reduced = vec![T::zero(); metadata.output.len];

    if should_parallelize_reduction(metadata.output.len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_iter_mut().enumerate().for_each(|(index, slot)| {
            let lane_offset = linear_offset(
                base_offset,
                &metadata.output.shape,
                &metadata.output.outer_strides,
                index,
            );
            *slot = sum_contiguous(contiguous_lane(data, lane_offset, metadata.axis_len));
        });
    } else {
        for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
            base_offset,
            &metadata.output.shape,
            &metadata.output.outer_strides,
        )) {
            *slot = sum_contiguous(contiguous_lane(data, lane_offset, metadata.axis_len));
        }
    }

    NDArray::from_shape_vec(metadata.output.shape, reduced)
}

fn sum_axis_strided<T: ElementwiseArithmetic>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>> {
    if simd::is_f32::<T>() {
        return sum_axis_strided_f32::<T>(simd::cast_slice(data), base_offset, metadata);
    }

    if simd::is_f64::<T>() {
        return sum_axis_strided_f64::<T>(simd::cast_slice(data), base_offset, metadata);
    }

    let mut reduced = vec![T::zero(); metadata.output.len];

    if should_parallelize_reduction(metadata.output.len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_iter_mut().enumerate().for_each(|(index, slot)| {
            let lane_offset = linear_offset(
                base_offset,
                &metadata.output.shape,
                &metadata.output.outer_strides,
                index,
            );
            *slot = sum_strided_lane(data, lane_offset, metadata.axis_len, metadata.axis_stride);
        });
    } else {
        for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
            base_offset,
            &metadata.output.shape,
            &metadata.output.outer_strides,
        )) {
            *slot = sum_strided_lane(data, lane_offset, metadata.axis_len, metadata.axis_stride);
        }
    }

    NDArray::from_shape_vec(metadata.output.shape, reduced)
}

fn prod_axis_contiguous<T: ElementwiseArithmetic>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>> {
    let mut reduced = vec![T::zero(); metadata.output.len];

    if should_parallelize_reduction(metadata.output.len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_iter_mut().enumerate().for_each(|(index, slot)| {
            let lane_offset = linear_offset(
                base_offset,
                &metadata.output.shape,
                &metadata.output.outer_strides,
                index,
            );
            *slot = prod_contiguous(contiguous_lane(data, lane_offset, metadata.axis_len));
        });
    } else {
        for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
            base_offset,
            &metadata.output.shape,
            &metadata.output.outer_strides,
        )) {
            *slot = prod_contiguous(contiguous_lane(data, lane_offset, metadata.axis_len));
        }
    }

    NDArray::from_shape_vec(metadata.output.shape, reduced)
}

fn prod_axis_strided<T: ElementwiseArithmetic>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>> {
    let mut reduced = vec![T::zero(); metadata.output.len];

    if should_parallelize_reduction(metadata.output.len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_iter_mut().enumerate().for_each(|(index, slot)| {
            let lane_offset = linear_offset(
                base_offset,
                &metadata.output.shape,
                &metadata.output.outer_strides,
                index,
            );
            *slot = prod_strided_lane(data, lane_offset, metadata.axis_len, metadata.axis_stride);
        });
    } else {
        for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
            base_offset,
            &metadata.output.shape,
            &metadata.output.outer_strides,
        )) {
            *slot = prod_strided_lane(data, lane_offset, metadata.axis_len, metadata.axis_stride);
        }
    }

    NDArray::from_shape_vec(metadata.output.shape, reduced)
}

fn min_axis_contiguous<T>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
{
    let mut reduced = vec![T::zero(); metadata.output.len];

    if should_parallelize_reduction(metadata.output.len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_iter_mut().enumerate().try_for_each(|(index, slot)| -> AtlasNdResult<()> {
            let lane_offset = linear_offset(
                base_offset,
                &metadata.output.shape,
                &metadata.output.outer_strides,
                index,
            );
            *slot = min_contiguous(contiguous_lane(data, lane_offset, metadata.axis_len), "min")?;
            Ok(())
        })?;
    } else {
        for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
            base_offset,
            &metadata.output.shape,
            &metadata.output.outer_strides,
        )) {
            *slot = min_contiguous(contiguous_lane(data, lane_offset, metadata.axis_len), "min")?;
        }
    }

    NDArray::from_shape_vec(metadata.output.shape, reduced)
}

fn min_axis_strided<T>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
{
    let mut reduced = vec![T::zero(); metadata.output.len];

    if should_parallelize_reduction(metadata.output.len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_iter_mut().enumerate().for_each(|(index, slot)| {
            let lane_offset = linear_offset(
                base_offset,
                &metadata.output.shape,
                &metadata.output.outer_strides,
                index,
            );
            *slot = min_strided_lane(data, lane_offset, metadata.axis_len, metadata.axis_stride);
        });
    } else {
        for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
            base_offset,
            &metadata.output.shape,
            &metadata.output.outer_strides,
        )) {
            *slot = min_strided_lane(data, lane_offset, metadata.axis_len, metadata.axis_stride);
        }
    }

    NDArray::from_shape_vec(metadata.output.shape, reduced)
}

fn max_axis_contiguous<T>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
{
    let mut reduced = vec![T::zero(); metadata.output.len];

    if should_parallelize_reduction(metadata.output.len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_iter_mut().enumerate().try_for_each(|(index, slot)| -> AtlasNdResult<()> {
            let lane_offset = linear_offset(
                base_offset,
                &metadata.output.shape,
                &metadata.output.outer_strides,
                index,
            );
            *slot = max_contiguous(contiguous_lane(data, lane_offset, metadata.axis_len), "max")?;
            Ok(())
        })?;
    } else {
        for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
            base_offset,
            &metadata.output.shape,
            &metadata.output.outer_strides,
        )) {
            *slot = max_contiguous(contiguous_lane(data, lane_offset, metadata.axis_len), "max")?;
        }
    }

    NDArray::from_shape_vec(metadata.output.shape, reduced)
}

fn max_axis_strided<T>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
{
    let mut reduced = vec![T::zero(); metadata.output.len];

    if should_parallelize_reduction(metadata.output.len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_iter_mut().enumerate().for_each(|(index, slot)| {
            let lane_offset = linear_offset(
                base_offset,
                &metadata.output.shape,
                &metadata.output.outer_strides,
                index,
            );
            *slot = max_strided_lane(data, lane_offset, metadata.axis_len, metadata.axis_stride);
        });
    } else {
        for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
            base_offset,
            &metadata.output.shape,
            &metadata.output.outer_strides,
        )) {
            *slot = max_strided_lane(data, lane_offset, metadata.axis_len, metadata.axis_stride);
        }
    }

    NDArray::from_shape_vec(metadata.output.shape, reduced)
}

pub(super) fn linear_offset(
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    mut linear_index: usize,
) -> usize {
    let mut offset = base_offset;

    for axis in (0..shape.len()).rev() {
        let index = linear_index % shape[axis];
        linear_index /= shape[axis];
        offset += index * strides[axis];
    }

    offset
}

pub(super) fn contiguous_lane<T>(data: &[T], lane_offset: usize, axis_len: usize) -> &[T] {
    &data[lane_offset..lane_offset + axis_len]
}

pub(super) fn contiguous_region<T>(data: &[T], base_offset: usize, len: usize) -> &[T] {
    &data[base_offset..base_offset + len]
}

fn sum_strided_lane<T: ElementwiseArithmetic>(
    data: &[T],
    lane_offset: usize,
    axis_len: usize,
    axis_stride: usize,
) -> T {
    let mut total = T::zero();
    let mut offset = lane_offset;

    for _ in 0..axis_len {
        total = total.elementwise_add(data[offset]);
        offset += axis_stride;
    }

    total
}

fn sum_axis_dense_contiguous_f32<T: ElementwiseArithmetic>(
    values: &[f32],
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>> {
    let mut reduced = vec![T::zero(); metadata.output.len];

    if should_parallelize_reduction(metadata.output.len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_chunks_mut(metadata.contiguous_inner_len).enumerate().for_each(
            |(outer, output_row)| {
                let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;

                for (inner, slot) in output_row.iter_mut().enumerate() {
                    *slot = simd::cast_value_exact(simd::compensated_sum_strided_f32(
                        values,
                        block_start + inner,
                        metadata.axis_len,
                        metadata.contiguous_inner_len,
                    ));
                }
            },
        );
    } else {
        for outer in 0..metadata.contiguous_outer_len {
            let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;
            let output_start = outer * metadata.contiguous_inner_len;

            for inner in 0..metadata.contiguous_inner_len {
                reduced[output_start + inner] =
                    simd::cast_value_exact(simd::compensated_sum_strided_f32(
                        values,
                        block_start + inner,
                        metadata.axis_len,
                        metadata.contiguous_inner_len,
                    ));
            }
        }
    }

    NDArray::from_shape_vec(metadata.output.shape, reduced)
}

fn sum_axis_dense_contiguous_f64<T: ElementwiseArithmetic>(
    values: &[f64],
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>> {
    let mut reduced = vec![T::zero(); metadata.output.len];

    if should_parallelize_reduction(metadata.output.len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_chunks_mut(metadata.contiguous_inner_len).enumerate().for_each(
            |(outer, output_row)| {
                let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;

                for (inner, slot) in output_row.iter_mut().enumerate() {
                    *slot = simd::cast_value_exact(simd::compensated_sum_strided_f64(
                        values,
                        block_start + inner,
                        metadata.axis_len,
                        metadata.contiguous_inner_len,
                    ));
                }
            },
        );
    } else {
        for outer in 0..metadata.contiguous_outer_len {
            let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;
            let output_start = outer * metadata.contiguous_inner_len;

            for inner in 0..metadata.contiguous_inner_len {
                reduced[output_start + inner] =
                    simd::cast_value_exact(simd::compensated_sum_strided_f64(
                        values,
                        block_start + inner,
                        metadata.axis_len,
                        metadata.contiguous_inner_len,
                    ));
            }
        }
    }

    NDArray::from_shape_vec(metadata.output.shape, reduced)
}

fn sum_axis_strided_f32<T: ElementwiseArithmetic>(
    data: &[f32],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>> {
    let mut reduced = vec![T::zero(); metadata.output.len];

    if should_parallelize_reduction(metadata.output.len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_iter_mut().enumerate().for_each(|(index, slot)| {
            let lane_offset = linear_offset(
                base_offset,
                &metadata.output.shape,
                &metadata.output.outer_strides,
                index,
            );
            *slot = simd::cast_value_exact(sum_strided_lane_f32(
                data,
                lane_offset,
                metadata.axis_len,
                metadata.axis_stride,
            ));
        });
    } else {
        for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
            base_offset,
            &metadata.output.shape,
            &metadata.output.outer_strides,
        )) {
            *slot = simd::cast_value_exact(sum_strided_lane_f32(
                data,
                lane_offset,
                metadata.axis_len,
                metadata.axis_stride,
            ));
        }
    }

    NDArray::from_shape_vec(metadata.output.shape, reduced)
}

fn sum_axis_strided_f64<T: ElementwiseArithmetic>(
    data: &[f64],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>> {
    let mut reduced = vec![T::zero(); metadata.output.len];

    if should_parallelize_reduction(metadata.output.len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_iter_mut().enumerate().for_each(|(index, slot)| {
            let lane_offset = linear_offset(
                base_offset,
                &metadata.output.shape,
                &metadata.output.outer_strides,
                index,
            );
            *slot = simd::cast_value_exact(sum_strided_lane_f64(
                data,
                lane_offset,
                metadata.axis_len,
                metadata.axis_stride,
            ));
        });
    } else {
        for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
            base_offset,
            &metadata.output.shape,
            &metadata.output.outer_strides,
        )) {
            *slot = simd::cast_value_exact(sum_strided_lane_f64(
                data,
                lane_offset,
                metadata.axis_len,
                metadata.axis_stride,
            ));
        }
    }

    NDArray::from_shape_vec(metadata.output.shape, reduced)
}

fn sum_strided_lane_f32(
    data: &[f32],
    lane_offset: usize,
    axis_len: usize,
    axis_stride: usize,
) -> f32 {
    simd::compensated_sum_strided_f32(data, lane_offset, axis_len, axis_stride)
}

fn sum_strided_lane_f64(
    data: &[f64],
    lane_offset: usize,
    axis_len: usize,
    axis_stride: usize,
) -> f64 {
    simd::compensated_sum_strided_f64(data, lane_offset, axis_len, axis_stride)
}

fn prod_strided_lane<T: ElementwiseArithmetic>(
    data: &[T],
    lane_offset: usize,
    axis_len: usize,
    axis_stride: usize,
) -> T {
    let mut total = T::one();
    let mut offset = lane_offset;

    for _ in 0..axis_len {
        total = total.elementwise_mul(data[offset]);
        offset += axis_stride;
    }

    total
}

fn min_strided_lane<T>(data: &[T], lane_offset: usize, axis_len: usize, axis_stride: usize) -> T
where
    T: Numeric + PartialOrd,
{
    let mut offset = lane_offset;
    let mut current = data[offset];
    offset += axis_stride;

    for _ in 1..axis_len {
        let value = data[offset];
        current = simd::min_propagating(current, value);
        offset += axis_stride;
    }

    current
}

fn max_strided_lane<T>(data: &[T], lane_offset: usize, axis_len: usize, axis_stride: usize) -> T
where
    T: Numeric + PartialOrd,
{
    let mut offset = lane_offset;
    let mut current = data[offset];
    offset += axis_stride;

    for _ in 1..axis_len {
        let value = data[offset];
        current = simd::max_propagating(current, value);
        offset += axis_stride;
    }

    current
}
