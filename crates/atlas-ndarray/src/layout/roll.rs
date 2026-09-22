use crate::{
    ArrayElement, AtlasNdResult, AxisIndex, NDArray, OperandMetadata, core::axis::normalize_axis,
    internal::value_iter, layout::element_count,
};

impl<T: ArrayElement> NDArray<T> {
    /// Returns an owned array with values rolled along the selected axis.
    pub fn roll<A: AxisIndex>(&self, shift: i64, axis: A) -> AtlasNdResult<Self> {
        roll_operand(self, shift, axis)
    }
}

impl<'a, T: ArrayElement> crate::ArrayView<'a, T> {
    /// Returns an owned array with values rolled along the selected axis.
    pub fn roll<A: AxisIndex>(&self, shift: i64, axis: A) -> AtlasNdResult<NDArray<T>> {
        roll_operand(self, shift, axis)
    }
}

fn roll_operand<T, O, A>(operand: &O, shift: i64, axis: A) -> AtlasNdResult<NDArray<T>>
where
    T: ArrayElement,
    O: OperandMetadata<T> + ?Sized,
    A: AxisIndex,
{
    let axis = normalize_axis(axis, operand.ndim())?;
    let shape = operand.shape();
    let axis_len = shape[axis];
    let data: Vec<_> =
        value_iter(operand.data(), operand.offset(), shape, operand.strides()).copied().collect();
    if data.is_empty() {
        return NDArray::from_shape_vec(shape.to_vec(), data);
    }

    let inner = element_count(&shape[axis + 1..]);
    let shift = normalized_shift(shift, axis_len);
    let mut rolled = Vec::with_capacity(data.len());

    for index in 0..data.len() {
        let outer = index / (axis_len * inner);
        let coordinate = index / inner % axis_len;
        let within = index % inner;
        let source_coordinate = (coordinate + axis_len - shift) % axis_len;
        let source = outer * axis_len * inner + source_coordinate * inner + within;
        rolled.push(data[source]);
    }

    NDArray::from_shape_vec(shape.to_vec(), rolled)
}

fn normalized_shift(shift: i64, len: usize) -> usize {
    usize::try_from((shift as i128).rem_euclid(len as i128))
        .expect("a normalized shift always fits usize")
}
