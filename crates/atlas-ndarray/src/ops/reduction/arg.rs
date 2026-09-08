use super::metadata::{AxisReductionMetadata, ReductionOperand, WholeReductionMetadata};
use crate::{
    AtlasNdError, AtlasNdResult, AxisIndex, NDArray, Numeric, OperandMetadata,
    internal::{for_each_value, offset_iter},
};

pub(super) fn argmin_all<T: Numeric + PartialOrd>(
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> AtlasNdResult<usize> {
    arg_all(data, offset, shape, strides, "argmin", replaces_min)
}

pub(super) fn argmax_all<T: Numeric + PartialOrd>(
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> AtlasNdResult<usize> {
    arg_all(data, offset, shape, strides, "argmax", replaces_max)
}

pub(super) fn argmin_axis<T: Numeric + PartialOrd, O: OperandMetadata<T> + ?Sized>(
    operand: &O,
    axis: impl AxisIndex,
    keepdims: bool,
) -> AtlasNdResult<NDArray<usize>> {
    arg_axis(ReductionOperand::new(operand), axis, keepdims, "argmin", replaces_min)
}

pub(super) fn argmax_axis<T: Numeric + PartialOrd, O: OperandMetadata<T> + ?Sized>(
    operand: &O,
    axis: impl AxisIndex,
    keepdims: bool,
) -> AtlasNdResult<NDArray<usize>> {
    arg_axis(ReductionOperand::new(operand), axis, keepdims, "argmax", replaces_max)
}

fn arg_all<T: Numeric + PartialOrd>(
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    op: &'static str,
    replaces: fn(T, T) -> bool,
) -> AtlasNdResult<usize> {
    WholeReductionMetadata::from_shape(shape).require_non_empty(op)?;
    let mut best = None;
    let mut index = 0;

    for_each_value(data, offset, shape, strides, |value| {
        best = Some(match best {
            Some((best_index, current)) if !replaces(current, *value) => (best_index, current),
            _ => (index, *value),
        });
        index += 1;
    });

    best.map(|(index, _)| index).ok_or(AtlasNdError::EmptyReduction { op })
}

fn arg_axis<T: Numeric + PartialOrd>(
    operand: ReductionOperand<'_, T>,
    axis: impl AxisIndex,
    keepdims: bool,
    op: &'static str,
    replaces: fn(T, T) -> bool,
) -> AtlasNdResult<NDArray<usize>> {
    let metadata = AxisReductionMetadata::new(operand.shape, operand.strides, axis, keepdims)?;
    metadata.require_non_empty(op)?;
    let mut indices = Vec::with_capacity(metadata.output.len);

    for lane_offset in
        offset_iter(operand.offset, &metadata.output.shape, &metadata.output.outer_strides)
    {
        let mut best_index = 0;
        let mut best = operand.data[lane_offset];

        for index in 1..metadata.axis_len {
            let value = operand.data[lane_offset + index * metadata.axis_stride];
            if replaces(best, value) {
                best = value;
                best_index = index;
            }
        }

        indices.push(best_index);
    }

    NDArray::from_shape_vec(metadata.output.shape, indices)
}

fn replaces_min<T: PartialOrd>(current: T, value: T) -> bool {
    current.partial_cmp(&current).is_some()
        && (value.partial_cmp(&value).is_none() || value < current)
}

fn replaces_max<T: PartialOrd>(current: T, value: T) -> bool {
    current.partial_cmp(&current).is_some()
        && (value.partial_cmp(&value).is_none() || value > current)
}
