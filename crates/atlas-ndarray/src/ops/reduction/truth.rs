use rayon::prelude::*;

use super::{
    axis::{contiguous_lane, contiguous_region, linear_offset},
    dispatch::should_parallelize_reduction,
    metadata::{AxisReductionMetadata, WholeReductionMetadata},
};
use crate::{
    AtlasNdResult, AxisIndex, NDArray,
    internal::{
        layout::{LayoutKind, dense_storage_slice},
        logical_span_iter, offset_iter,
    },
};

pub(super) fn all_all(data: &[bool], offset: usize, shape: &[usize], strides: &[usize]) -> bool {
    let metadata = WholeReductionMetadata::from_shape(shape);
    if metadata.is_empty() {
        return true;
    }

    if let Some(values) = dense_storage_slice(data, offset, shape, strides) {
        return all_contiguous(values);
    }

    logical_span_iter(data, offset, shape, strides).all(all_contiguous)
}

pub(super) fn any_all(data: &[bool], offset: usize, shape: &[usize], strides: &[usize]) -> bool {
    let metadata = WholeReductionMetadata::from_shape(shape);
    if metadata.is_empty() {
        return false;
    }

    if let Some(values) = dense_storage_slice(data, offset, shape, strides) {
        return any_contiguous(values);
    }

    logical_span_iter(data, offset, shape, strides).any(any_contiguous)
}

pub(super) fn count_true_all(
    data: &[bool],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> usize {
    if let Some(values) = dense_storage_slice(data, offset, shape, strides) {
        return values.iter().filter(|&&value| value).count();
    }

    logical_span_iter(data, offset, shape, strides)
        .map(|span| span.iter().filter(|&&value| value).count())
        .sum()
}

pub(super) fn all_axis_impl(
    data: &[bool],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<bool>> {
    let metadata = AxisReductionMetadata::new(shape, strides, axis, false)?;
    dispatch_all_axis(data, base_offset, metadata)
}

pub(super) fn all_axis_keepdims_impl(
    data: &[bool],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<bool>> {
    let metadata = AxisReductionMetadata::new(shape, strides, axis, true)?;
    dispatch_all_axis(data, base_offset, metadata)
}

fn dispatch_all_axis(
    data: &[bool],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<bool>> {
    match (metadata.source_layout, metadata.axis_layout) {
        (_, LayoutKind::Contiguous) => all_axis_contiguous(data, base_offset, metadata),
        (LayoutKind::Contiguous, LayoutKind::Strided) => {
            all_axis_dense_contiguous(data, base_offset, metadata)
        }
        (LayoutKind::Strided, LayoutKind::Strided) => all_axis_strided(data, base_offset, metadata),
    }
}

pub(super) fn any_axis_impl(
    data: &[bool],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<bool>> {
    let metadata = AxisReductionMetadata::new(shape, strides, axis, false)?;
    dispatch_any_axis(data, base_offset, metadata)
}

pub(super) fn any_axis_keepdims_impl(
    data: &[bool],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<bool>> {
    let metadata = AxisReductionMetadata::new(shape, strides, axis, true)?;
    dispatch_any_axis(data, base_offset, metadata)
}

fn dispatch_any_axis(
    data: &[bool],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<bool>> {
    match (metadata.source_layout, metadata.axis_layout) {
        (_, LayoutKind::Contiguous) => any_axis_contiguous(data, base_offset, metadata),
        (LayoutKind::Contiguous, LayoutKind::Strided) => {
            any_axis_dense_contiguous(data, base_offset, metadata)
        }
        (LayoutKind::Strided, LayoutKind::Strided) => any_axis_strided(data, base_offset, metadata),
    }
}

fn all_contiguous(values: &[bool]) -> bool {
    if should_parallelize_reduction(values.len()) {
        return values
            .par_chunks(super::dispatch::parallel_reduction_chunk_len())
            .all(all_contiguous);
    }

    values.iter().copied().all(|value| value)
}

fn any_contiguous(values: &[bool]) -> bool {
    if should_parallelize_reduction(values.len()) {
        return values
            .par_chunks(super::dispatch::parallel_reduction_chunk_len())
            .any(any_contiguous);
    }

    values.iter().copied().any(|value| value)
}

fn all_axis_dense_contiguous(
    data: &[bool],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<bool>> {
    let values =
        contiguous_region(data, base_offset, metadata.output.len.saturating_mul(metadata.axis_len));
    let mut reduced = vec![true; metadata.output.len];

    if should_parallelize_reduction(metadata.output.len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_chunks_mut(metadata.contiguous_inner_len).enumerate().for_each(
            |(outer, output_row)| {
                let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;

                for (inner, slot) in output_row.iter_mut().enumerate() {
                    let mut total = true;
                    let mut offset = block_start + inner;

                    for _ in 0..metadata.axis_len {
                        total &= values[offset];
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
                let mut total = true;
                let mut offset = block_start + inner;

                for _ in 0..metadata.axis_len {
                    total &= values[offset];
                    offset += metadata.contiguous_inner_len;
                }

                reduced[output_start + inner] = total;
            }
        }
    }

    NDArray::from_shape_vec(metadata.output.shape, reduced)
}

fn any_axis_dense_contiguous(
    data: &[bool],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<bool>> {
    let values =
        contiguous_region(data, base_offset, metadata.output.len.saturating_mul(metadata.axis_len));
    let mut reduced = vec![false; metadata.output.len];

    if should_parallelize_reduction(metadata.output.len.saturating_mul(metadata.axis_len))
        && !reduced.is_empty()
    {
        reduced.par_chunks_mut(metadata.contiguous_inner_len).enumerate().for_each(
            |(outer, output_row)| {
                let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;

                for (inner, slot) in output_row.iter_mut().enumerate() {
                    let mut total = false;
                    let mut offset = block_start + inner;

                    for _ in 0..metadata.axis_len {
                        total |= values[offset];
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
                let mut total = false;
                let mut offset = block_start + inner;

                for _ in 0..metadata.axis_len {
                    total |= values[offset];
                    offset += metadata.contiguous_inner_len;
                }

                reduced[output_start + inner] = total;
            }
        }
    }

    NDArray::from_shape_vec(metadata.output.shape, reduced)
}

fn all_axis_contiguous(
    data: &[bool],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<bool>> {
    let mut reduced = vec![true; metadata.output.len];

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
            *slot = all_contiguous(contiguous_lane(data, lane_offset, metadata.axis_len));
        });
    } else {
        for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
            base_offset,
            &metadata.output.shape,
            &metadata.output.outer_strides,
        )) {
            *slot = all_contiguous(contiguous_lane(data, lane_offset, metadata.axis_len));
        }
    }

    NDArray::from_shape_vec(metadata.output.shape, reduced)
}

fn any_axis_contiguous(
    data: &[bool],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<bool>> {
    let mut reduced = vec![false; metadata.output.len];

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
            *slot = any_contiguous(contiguous_lane(data, lane_offset, metadata.axis_len));
        });
    } else {
        for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
            base_offset,
            &metadata.output.shape,
            &metadata.output.outer_strides,
        )) {
            *slot = any_contiguous(contiguous_lane(data, lane_offset, metadata.axis_len));
        }
    }

    NDArray::from_shape_vec(metadata.output.shape, reduced)
}

fn all_axis_strided(
    data: &[bool],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<bool>> {
    let mut reduced = vec![true; metadata.output.len];

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
            *slot = all_strided_lane(data, lane_offset, metadata.axis_len, metadata.axis_stride);
        });
    } else {
        for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
            base_offset,
            &metadata.output.shape,
            &metadata.output.outer_strides,
        )) {
            *slot = all_strided_lane(data, lane_offset, metadata.axis_len, metadata.axis_stride);
        }
    }

    NDArray::from_shape_vec(metadata.output.shape, reduced)
}

fn any_axis_strided(
    data: &[bool],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<bool>> {
    let mut reduced = vec![false; metadata.output.len];

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
            *slot = any_strided_lane(data, lane_offset, metadata.axis_len, metadata.axis_stride);
        });
    } else {
        for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
            base_offset,
            &metadata.output.shape,
            &metadata.output.outer_strides,
        )) {
            *slot = any_strided_lane(data, lane_offset, metadata.axis_len, metadata.axis_stride);
        }
    }

    NDArray::from_shape_vec(metadata.output.shape, reduced)
}

fn all_strided_lane(
    data: &[bool],
    lane_offset: usize,
    axis_len: usize,
    axis_stride: usize,
) -> bool {
    let mut total = true;
    let mut offset = lane_offset;

    for _ in 0..axis_len {
        total &= data[offset];
        offset += axis_stride;
    }

    total
}

fn any_strided_lane(
    data: &[bool],
    lane_offset: usize,
    axis_len: usize,
    axis_stride: usize,
) -> bool {
    let mut total = false;
    let mut offset = lane_offset;

    for _ in 0..axis_len {
        total |= data[offset];
        offset += axis_stride;
    }

    total
}
