use crate::{
    ArrayElement, AtlasNdResult, NDArray, OperandMetadata, internal::value_iter, view::ArrayView,
};

impl<T> NDArray<T>
where
    T: ArrayElement + Default + PartialEq,
{
    /// Returns logical coordinates of values unequal to zero (or false for boolean arrays).
    pub fn nonzero_indices(&self) -> AtlasNdResult<NDArray<usize>> {
        nonzero_indices_operand(self)
    }
}

impl<T> ArrayView<'_, T>
where
    T: ArrayElement + Default + PartialEq,
{
    /// Returns logical coordinates of values unequal to zero (or false for boolean arrays).
    pub fn nonzero_indices(&self) -> AtlasNdResult<NDArray<usize>> {
        nonzero_indices_operand(self)
    }
}

fn nonzero_indices_operand<T, O>(operand: &O) -> AtlasNdResult<NDArray<usize>>
where
    T: ArrayElement + Default + PartialEq,
    O: OperandMetadata<T> + ?Sized,
{
    let shape = operand.shape();
    let mut index = vec![0; shape.len()];
    let mut indices = Vec::new();
    let mut matches = 0;
    let zero = T::default();

    for (linear_index, value) in
        value_iter(operand.data(), operand.offset(), shape, operand.strides()).enumerate()
    {
        if *value != zero {
            logical_index(linear_index, shape, &mut index);
            indices.extend_from_slice(&index);
            matches += 1;
        }
    }

    NDArray::from_shape_vec([matches, shape.len()], indices)
}

fn logical_index(mut linear_index: usize, shape: &[usize], index: &mut [usize]) {
    for axis in (0..shape.len()).rev() {
        index[axis] = linear_index % shape[axis];
        linear_index /= shape[axis];
    }
}
