use crate::{
    ArrayElement, AtlasNdResult, AxisIndex, NDArray, OperandMetadata, core::axis::normalize_axis,
    internal::value_iter, layout::element_count,
};

impl<T: ArrayElement> NDArray<T> {
    /// Returns an owned array with all axes reversed.
    pub fn flip(&self) -> Self {
        flip_operand(self)
    }

    /// Returns an owned array with the selected axis reversed.
    pub fn flip_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<Self> {
        flip_axis_operand(self, axis)
    }
}

impl<'a, T: ArrayElement> crate::ArrayView<'a, T> {
    /// Returns an owned array with all axes reversed.
    pub fn flip(&self) -> NDArray<T> {
        flip_operand(self)
    }

    /// Returns an owned array with the selected axis reversed.
    pub fn flip_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<T>> {
        flip_axis_operand(self, axis)
    }
}

fn flip_operand<T, O>(operand: &O) -> NDArray<T>
where
    T: ArrayElement,
    O: OperandMetadata<T> + ?Sized,
{
    let mut data: Vec<_> =
        value_iter(operand.data(), operand.offset(), operand.shape(), operand.strides())
            .copied()
            .collect();
    data.reverse();

    NDArray::from_shape_vec(operand.shape().to_vec(), data)
        .expect("flipping preserves ndarray invariants")
}

fn flip_axis_operand<T, O, A>(operand: &O, axis: A) -> AtlasNdResult<NDArray<T>>
where
    T: ArrayElement,
    O: OperandMetadata<T> + ?Sized,
    A: AxisIndex,
{
    let axis = normalize_axis(axis, operand.ndim())?;
    let shape = operand.shape();
    let axis_len = shape[axis];
    let inner = element_count(&shape[axis + 1..]);
    let data: Vec<_> =
        value_iter(operand.data(), operand.offset(), shape, operand.strides()).copied().collect();
    let mut flipped = Vec::with_capacity(data.len());

    for index in 0..data.len() {
        let outer = index / (axis_len * inner);
        let coordinate = index / inner % axis_len;
        let within = index % inner;
        let source = outer * axis_len * inner + (axis_len - 1 - coordinate) * inner + within;
        flipped.push(data[source]);
    }

    NDArray::from_shape_vec(shape.to_vec(), flipped)
}
