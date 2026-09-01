use crate::{
    ArrayElement, AtlasNdError, AtlasNdResult, AxisIndex, NDArray, OperandMetadata,
    checked_compute_strides, checked_element_count, core::axis::normalize_axis,
    internal::value_iter, view::ArrayView,
};

impl<T: ArrayElement> NDArray<T> {
    /// Returns an owned array with each logical element block repeated along `axis`.
    pub fn repeat<A: AxisIndex>(&self, repeats: usize, axis: A) -> AtlasNdResult<Self> {
        repeat_operand(self, repeats, axis)
    }

    /// Returns an owned array tiled according to `repetitions`.
    pub fn tile<R: AsRef<[usize]>>(&self, repetitions: R) -> AtlasNdResult<Self> {
        tile_operand(self, repetitions.as_ref())
    }
}

impl<'a, T: ArrayElement> ArrayView<'a, T> {
    /// Returns an owned array with each logical element block repeated along `axis`.
    pub fn repeat<A: AxisIndex>(&self, repeats: usize, axis: A) -> AtlasNdResult<NDArray<T>> {
        repeat_operand(self, repeats, axis)
    }

    /// Returns an owned array tiled according to `repetitions`.
    pub fn tile<R: AsRef<[usize]>>(&self, repetitions: R) -> AtlasNdResult<NDArray<T>> {
        tile_operand(self, repetitions.as_ref())
    }
}

fn repeat_operand<T, O, A>(operand: &O, repeats: usize, axis: A) -> AtlasNdResult<NDArray<T>>
where
    T: ArrayElement,
    O: OperandMetadata<T> + ?Sized,
    A: AxisIndex,
{
    let axis = normalize_axis(axis, operand.ndim())?;
    let mut shape = operand.shape().to_vec();
    shape[axis] = shape[axis]
        .checked_mul(repeats)
        .ok_or_else(|| AtlasNdError::ShapeOverflow { op: "repeat", shape: shape.clone() })?;

    let inner = checked_element_count(&operand.shape()[axis + 1..])?;
    let outer = checked_element_count(&operand.shape()[..axis])?;
    let axis_len = operand.shape()[axis];
    let values: Vec<_> =
        value_iter(operand.data(), operand.offset(), operand.shape(), operand.strides())
            .copied()
            .collect();
    let mut data = Vec::with_capacity(checked_element_count(&shape)?);

    for outer_index in 0..outer {
        let block_start = outer_index * axis_len * inner;
        for axis_index in 0..axis_len {
            let value_start = block_start + axis_index * inner;
            let values = &values[value_start..value_start + inner];
            for _ in 0..repeats {
                data.extend_from_slice(values);
            }
        }
    }

    NDArray::from_shape_vec(shape, data)
}

fn tile_operand<T, O>(operand: &O, repetitions: &[usize]) -> AtlasNdResult<NDArray<T>>
where
    T: ArrayElement,
    O: OperandMetadata<T> + ?Sized,
{
    if repetitions.is_empty() {
        return Err(AtlasNdError::InvalidArgument {
            op: "tile",
            reason: "repetitions must not be empty",
        });
    }

    let ndim = operand.ndim().max(repetitions.len());
    let mut input_shape = vec![1; ndim - operand.ndim()];
    input_shape.extend_from_slice(operand.shape());
    let mut tiled_shape = Vec::with_capacity(ndim);
    for (dimension, repetition) in input_shape
        .iter()
        .zip(std::iter::repeat_n(1, ndim - repetitions.len()).chain(repetitions.iter().copied()))
    {
        tiled_shape.push(dimension.checked_mul(repetition).ok_or_else(|| {
            AtlasNdError::ShapeOverflow { op: "tile", shape: input_shape.clone() }
        })?);
    }

    let output_len = checked_element_count(&tiled_shape)?;
    if output_len == 0 {
        return NDArray::from_shape_vec(tiled_shape, Vec::new());
    }

    let input_strides = checked_compute_strides(&input_shape)?;
    let output_strides = checked_compute_strides(&tiled_shape)?;
    let values: Vec<_> =
        value_iter(operand.data(), operand.offset(), operand.shape(), operand.strides())
            .copied()
            .collect();
    let mut data = Vec::with_capacity(output_len);

    for output_index in 0..output_len {
        let mut source_index = 0;
        for axis in 0..ndim {
            let coordinate = output_index / output_strides[axis] % tiled_shape[axis];
            source_index += (coordinate % input_shape[axis]) * input_strides[axis];
        }
        data.push(values[source_index]);
    }

    NDArray::from_shape_vec(tiled_shape, data)
}
