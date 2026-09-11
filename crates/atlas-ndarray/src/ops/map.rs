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
