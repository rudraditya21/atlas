use num_traits::ToPrimitive;
use rayon::prelude::*;

use crate::{
    AtlasNdError, AtlasNdResult, NDArray, Numeric,
    internal::{offset_iter, simd, try_for_each_value},
};

use super::{
    axis::{AxisReductionMetadata, contiguous_lane, linear_offset},
    dispatch::{
        ensure_non_empty_reduction, parallel_reduction_chunk_len, should_parallelize_reduction,
    },
};

pub(super) fn mean_contiguous<T>(values: &[T], op: &'static str) -> AtlasNdResult<f64>
where
    T: Numeric + ToPrimitive,
{
    if should_parallelize_reduction(values.len()) {
        if values.is_empty() {
            return Err(AtlasNdError::EmptyReduction { op });
        }

        let partials: Vec<AtlasNdResult<f64>> = values
            .par_chunks(parallel_reduction_chunk_len())
            .map(|chunk| sum_chunk_as_f64(chunk, op))
            .collect();
        let total =
            partials.into_iter().try_fold(0.0_f64, |total, partial| Ok(total + partial?))?;

        return Ok(total / values.len() as f64);
    }

    simd::mean_contiguous(values, op)
}

pub(super) fn mean_strided_lane<T>(
    data: &[T],
    lane_offset: usize,
    axis_len: usize,
    axis_stride: usize,
    op: &'static str,
) -> AtlasNdResult<f64>
where
    T: Numeric + ToPrimitive,
{
    let mut total = 0.0_f64;
    let mut offset = lane_offset;

    for _ in 0..axis_len {
        total += data[offset].to_f64().ok_or(AtlasNdError::NumericConversionFailed { op })?;
        offset += axis_stride;
    }

    Ok(total / axis_len as f64)
}

pub(super) fn mean_strided_lane_f32(
    data: &[f32],
    lane_offset: usize,
    axis_len: usize,
    axis_stride: usize,
) -> f64 {
    let mut total = 0.0_f64;
    let mut offset = lane_offset;

    for _ in 0..axis_len {
        total += f64::from(data[offset]);
        offset += axis_stride;
    }

    total / axis_len as f64
}

pub(super) fn mean_strided_lane_f64(
    data: &[f64],
    lane_offset: usize,
    axis_len: usize,
    axis_stride: usize,
) -> f64 {
    let mut total = 0.0_f64;
    let mut offset = lane_offset;

    for _ in 0..axis_len {
        total += data[offset];
        offset += axis_stride;
    }

    total / axis_len as f64
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
    let len = crate::layout::element_count(shape);
    ensure_non_empty_reduction(len, "mean")?;

    if let Some(values) = crate::internal::layout::dense_storage_slice(data, offset, shape, strides)
    {
        return mean_contiguous(values, "mean");
    }

    if simd::is_f32::<T>() {
        return mean_all_f32(simd::cast_slice(data), offset, shape, strides, len);
    }

    if simd::is_f64::<T>() {
        return mean_all_f64(simd::cast_slice(data), offset, shape, strides, len);
    }

    let mut total = 0.0_f64;
    try_for_each_value(data, offset, shape, strides, |value| {
        total += value.to_f64().ok_or(AtlasNdError::NumericConversionFailed { op: "mean" })?;
        Ok(())
    })?;

    Ok(total / len as f64)
}

pub(super) fn mean_axis_dense_contiguous<T>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
{
    let values = super::axis::contiguous_region(
        data,
        base_offset,
        metadata.output_len.saturating_mul(metadata.axis_len),
    );

    if simd::is_f32::<T>() {
        return mean_axis_dense_contiguous_f32(simd::cast_slice(values), metadata);
    }

    if simd::is_f64::<T>() {
        return mean_axis_dense_contiguous_f64(simd::cast_slice(values), metadata);
    }

    let mut reduced = vec![0.0_f64; metadata.output_len];

    if should_parallelize_reduction(metadata.output_len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_chunks_mut(metadata.contiguous_inner_len).enumerate().try_for_each(
            |(outer, output_row)| -> AtlasNdResult<()> {
                let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;

                for (inner, slot) in output_row.iter_mut().enumerate() {
                    let mut total = 0.0_f64;
                    let mut offset = block_start + inner;

                    for _ in 0..metadata.axis_len {
                        total += values[offset]
                            .to_f64()
                            .ok_or(AtlasNdError::NumericConversionFailed { op: "mean" })?;
                        offset += metadata.contiguous_inner_len;
                    }

                    *slot = total / metadata.axis_len as f64;
                }

                Ok(())
            },
        )?;
    } else {
        for outer in 0..metadata.contiguous_outer_len {
            let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;
            let output_start = outer * metadata.contiguous_inner_len;

            for inner in 0..metadata.contiguous_inner_len {
                let mut total = 0.0_f64;
                let mut offset = block_start + inner;

                for _ in 0..metadata.axis_len {
                    total += values[offset]
                        .to_f64()
                        .ok_or(AtlasNdError::NumericConversionFailed { op: "mean" })?;
                    offset += metadata.contiguous_inner_len;
                }

                reduced[output_start + inner] = total / metadata.axis_len as f64;
            }
        }
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

pub(super) fn mean_axis_contiguous<T>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
{
    if simd::is_f32::<T>() {
        return mean_axis_contiguous_f32(simd::cast_slice(data), base_offset, metadata);
    }

    if simd::is_f64::<T>() {
        return mean_axis_contiguous_f64(simd::cast_slice(data), base_offset, metadata);
    }

    let mut reduced = vec![0.0_f64; metadata.output_len];

    if should_parallelize_reduction(metadata.output_len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_iter_mut().enumerate().try_for_each(|(index, slot)| -> AtlasNdResult<()> {
            let lane_offset =
                linear_offset(base_offset, &metadata.output_shape, &metadata.outer_strides, index);
            *slot = mean_contiguous(contiguous_lane(data, lane_offset, metadata.axis_len), "mean")?;
            Ok(())
        })?;
    } else {
        for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
            base_offset,
            &metadata.output_shape,
            &metadata.outer_strides,
        )) {
            *slot = mean_contiguous(contiguous_lane(data, lane_offset, metadata.axis_len), "mean")?;
        }
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

pub(super) fn mean_axis_strided<T>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
{
    if simd::is_f32::<T>() {
        return mean_axis_strided_f32(simd::cast_slice(data), base_offset, metadata);
    }

    if simd::is_f64::<T>() {
        return mean_axis_strided_f64(simd::cast_slice(data), base_offset, metadata);
    }

    let mut reduced = vec![0.0_f64; metadata.output_len];

    if should_parallelize_reduction(metadata.output_len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_iter_mut().enumerate().try_for_each(|(index, slot)| -> AtlasNdResult<()> {
            let lane_offset =
                linear_offset(base_offset, &metadata.output_shape, &metadata.outer_strides, index);
            *slot = mean_strided_lane(
                data,
                lane_offset,
                metadata.axis_len,
                metadata.axis_stride,
                "mean",
            )?;
            Ok(())
        })?;
    } else {
        for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
            base_offset,
            &metadata.output_shape,
            &metadata.outer_strides,
        )) {
            *slot = mean_strided_lane(
                data,
                lane_offset,
                metadata.axis_len,
                metadata.axis_stride,
                "mean",
            )?;
        }
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn sum_chunk_as_f64<T>(values: &[T], op: &'static str) -> AtlasNdResult<f64>
where
    T: Numeric + ToPrimitive,
{
    if simd::is_f32::<T>() {
        return Ok(simd::cast_slice::<T, f32>(values).iter().copied().map(f64::from).sum());
    }

    if simd::is_f64::<T>() {
        return Ok(simd::cast_slice::<T, f64>(values).iter().copied().sum());
    }

    values.iter().try_fold(0.0_f64, |total, value| {
        Ok(total + value.to_f64().ok_or(AtlasNdError::NumericConversionFailed { op })?)
    })
}

fn mean_all_f32(
    data: &[f32],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    len: usize,
) -> AtlasNdResult<f64> {
    let mut total = 0.0_f64;
    try_for_each_value(data, offset, shape, strides, |value| {
        total += f64::from(*value);
        Ok::<(), AtlasNdError>(())
    })?;
    Ok(total / len as f64)
}

fn mean_all_f64(
    data: &[f64],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    len: usize,
) -> AtlasNdResult<f64> {
    let mut total = 0.0_f64;
    try_for_each_value(data, offset, shape, strides, |value| {
        total += *value;
        Ok::<(), AtlasNdError>(())
    })?;
    Ok(total / len as f64)
}

fn mean_axis_dense_contiguous_f32(
    values: &[f32],
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<f64>> {
    let mut reduced = vec![0.0_f64; metadata.output_len];

    if should_parallelize_reduction(metadata.output_len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_chunks_mut(metadata.contiguous_inner_len).enumerate().for_each(
            |(outer, output_row)| {
                let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;

                for (inner, slot) in output_row.iter_mut().enumerate() {
                    let mut total = 0.0_f64;
                    let mut offset = block_start + inner;

                    for _ in 0..metadata.axis_len {
                        total += f64::from(values[offset]);
                        offset += metadata.contiguous_inner_len;
                    }

                    *slot = total / metadata.axis_len as f64;
                }
            },
        );
    } else {
        for outer in 0..metadata.contiguous_outer_len {
            let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;
            let output_start = outer * metadata.contiguous_inner_len;

            for inner in 0..metadata.contiguous_inner_len {
                let mut total = 0.0_f64;
                let mut offset = block_start + inner;

                for _ in 0..metadata.axis_len {
                    total += f64::from(values[offset]);
                    offset += metadata.contiguous_inner_len;
                }

                reduced[output_start + inner] = total / metadata.axis_len as f64;
            }
        }
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn mean_axis_dense_contiguous_f64(
    values: &[f64],
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<f64>> {
    let mut reduced = vec![0.0_f64; metadata.output_len];

    if should_parallelize_reduction(metadata.output_len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_chunks_mut(metadata.contiguous_inner_len).enumerate().for_each(
            |(outer, output_row)| {
                let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;

                for (inner, slot) in output_row.iter_mut().enumerate() {
                    let mut total = 0.0_f64;
                    let mut offset = block_start + inner;

                    for _ in 0..metadata.axis_len {
                        total += values[offset];
                        offset += metadata.contiguous_inner_len;
                    }

                    *slot = total / metadata.axis_len as f64;
                }
            },
        );
    } else {
        for outer in 0..metadata.contiguous_outer_len {
            let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;
            let output_start = outer * metadata.contiguous_inner_len;

            for inner in 0..metadata.contiguous_inner_len {
                let mut total = 0.0_f64;
                let mut offset = block_start + inner;

                for _ in 0..metadata.axis_len {
                    total += values[offset];
                    offset += metadata.contiguous_inner_len;
                }

                reduced[output_start + inner] = total / metadata.axis_len as f64;
            }
        }
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn mean_axis_contiguous_f32(
    data: &[f32],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<f64>> {
    let mut reduced = vec![0.0_f64; metadata.output_len];

    if should_parallelize_reduction(metadata.output_len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_iter_mut().enumerate().for_each(|(index, slot)| {
            let lane_offset =
                linear_offset(base_offset, &metadata.output_shape, &metadata.outer_strides, index);
            let lane = contiguous_lane(data, lane_offset, metadata.axis_len);
            *slot = lane.iter().copied().map(f64::from).sum::<f64>() / metadata.axis_len as f64;
        });
    } else {
        for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
            base_offset,
            &metadata.output_shape,
            &metadata.outer_strides,
        )) {
            let lane = contiguous_lane(data, lane_offset, metadata.axis_len);
            *slot = lane.iter().copied().map(f64::from).sum::<f64>() / metadata.axis_len as f64;
        }
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn mean_axis_contiguous_f64(
    data: &[f64],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<f64>> {
    let mut reduced = vec![0.0_f64; metadata.output_len];

    if should_parallelize_reduction(metadata.output_len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_iter_mut().enumerate().for_each(|(index, slot)| {
            let lane_offset =
                linear_offset(base_offset, &metadata.output_shape, &metadata.outer_strides, index);
            let lane = contiguous_lane(data, lane_offset, metadata.axis_len);
            *slot = lane.iter().copied().sum::<f64>() / metadata.axis_len as f64;
        });
    } else {
        for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
            base_offset,
            &metadata.output_shape,
            &metadata.outer_strides,
        )) {
            let lane = contiguous_lane(data, lane_offset, metadata.axis_len);
            *slot = lane.iter().copied().sum::<f64>() / metadata.axis_len as f64;
        }
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn mean_axis_strided_f32(
    data: &[f32],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<f64>> {
    let mut reduced = vec![0.0_f64; metadata.output_len];

    if should_parallelize_reduction(metadata.output_len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_iter_mut().enumerate().for_each(|(index, slot)| {
            let lane_offset =
                linear_offset(base_offset, &metadata.output_shape, &metadata.outer_strides, index);
            *slot =
                mean_strided_lane_f32(data, lane_offset, metadata.axis_len, metadata.axis_stride);
        });
    } else {
        for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
            base_offset,
            &metadata.output_shape,
            &metadata.outer_strides,
        )) {
            *slot =
                mean_strided_lane_f32(data, lane_offset, metadata.axis_len, metadata.axis_stride);
        }
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn mean_axis_strided_f64(
    data: &[f64],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<f64>> {
    let mut reduced = vec![0.0_f64; metadata.output_len];

    if should_parallelize_reduction(metadata.output_len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_iter_mut().enumerate().for_each(|(index, slot)| {
            let lane_offset =
                linear_offset(base_offset, &metadata.output_shape, &metadata.outer_strides, index);
            *slot =
                mean_strided_lane_f64(data, lane_offset, metadata.axis_len, metadata.axis_stride);
        });
    } else {
        for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
            base_offset,
            &metadata.output_shape,
            &metadata.outer_strides,
        )) {
            *slot =
                mean_strided_lane_f64(data, lane_offset, metadata.axis_len, metadata.axis_stride);
        }
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}
