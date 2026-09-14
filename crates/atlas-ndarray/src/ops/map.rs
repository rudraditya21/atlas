use crate::{ArrayElement, NDArray, OperandMetadata, internal::value_iter, view::ArrayView};

impl<T: ArrayElement> NDArray<T> {
    /// Maps each value into a new contiguous array with the same shape.
    pub fn map<U, F>(&self, op: F) -> NDArray<U>
    where
        U: ArrayElement,
        F: FnMut(T) -> U,
    {
        map_operand(self, op)
    }

    /// Maps each value and its logical multidimensional index into a contiguous array.
    pub fn map_indexed<U, F>(&self, op: F) -> NDArray<U>
    where
        U: ArrayElement,
        F: FnMut(&[usize], T) -> U,
    {
        map_indexed_operand(self, op)
    }
}

impl<T: ArrayElement> ArrayView<'_, T> {
    /// Maps logical view values into a new contiguous array with the same shape.
    pub fn map<U, F>(&self, op: F) -> NDArray<U>
    where
        U: ArrayElement,
        F: FnMut(T) -> U,
    {
        map_operand(self, op)
    }

    /// Maps each logical view value and its multidimensional index into a contiguous array.
    pub fn map_indexed<U, F>(&self, op: F) -> NDArray<U>
    where
        U: ArrayElement,
        F: FnMut(&[usize], T) -> U,
    {
        map_indexed_operand(self, op)
    }
}

fn map_operand<T, U, O, F>(operand: &O, op: F) -> NDArray<U>
where
    T: ArrayElement,
    U: ArrayElement,
    O: OperandMetadata<T> + ?Sized,
    F: FnMut(T) -> U,
{
    let data = value_iter(operand.data(), operand.offset(), operand.shape(), operand.strides())
        .copied()
        .map(op)
        .collect();

    NDArray::from_row_major_parts(operand.shape().to_vec(), data)
        .expect("mapping preserves ndarray invariants")
}

fn map_indexed_operand<T, U, O, F>(operand: &O, mut op: F) -> NDArray<U>
where
    T: ArrayElement,
    U: ArrayElement,
    O: OperandMetadata<T> + ?Sized,
    F: FnMut(&[usize], T) -> U,
{
    let mut index = vec![0; operand.shape().len()];
    let mut data = Vec::new();
    for (linear_index, value) in
        value_iter(operand.data(), operand.offset(), operand.shape(), operand.strides())
            .copied()
            .enumerate()
    {
        logical_index(linear_index, operand.shape(), &mut index);
        data.push(op(&index, value));
    }

    NDArray::from_row_major_parts(operand.shape().to_vec(), data)
        .expect("indexed mapping preserves ndarray invariants")
}

fn logical_index(mut linear_index: usize, shape: &[usize], index: &mut [usize]) {
    for axis in (0..shape.len()).rev() {
        index[axis] = linear_index % shape[axis];
        linear_index /= shape[axis];
    }
}
