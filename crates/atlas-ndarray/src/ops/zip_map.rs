use crate::{
    ArrayElement, AtlasNdError, AtlasNdResult, NDArray, OperandMetadata, internal::value_iter,
    view::ArrayView,
};

impl<T: ArrayElement> NDArray<T> {
    /// Maps pairs of logical values from identically shaped operands into a contiguous array.
    pub fn zip_map<U, V, O, F>(&self, rhs: &O, op: F) -> AtlasNdResult<NDArray<V>>
    where
        U: ArrayElement,
        V: ArrayElement,
        O: OperandMetadata<U> + ?Sized,
        F: FnMut(T, U) -> V,
    {
        zip_map_operands(self, rhs, op)
    }
}

impl<T: ArrayElement> ArrayView<'_, T> {
    /// Maps pairs of logical values from identically shaped operands into a contiguous array.
    pub fn zip_map<U, V, O, F>(&self, rhs: &O, op: F) -> AtlasNdResult<NDArray<V>>
    where
        U: ArrayElement,
        V: ArrayElement,
        O: OperandMetadata<U> + ?Sized,
        F: FnMut(T, U) -> V,
    {
        zip_map_operands(self, rhs, op)
    }
}

fn zip_map_operands<T, U, V, L, R, F>(lhs: &L, rhs: &R, mut op: F) -> AtlasNdResult<NDArray<V>>
where
    T: ArrayElement,
    U: ArrayElement,
    V: ArrayElement,
    L: OperandMetadata<T> + ?Sized,
    R: OperandMetadata<U> + ?Sized,
    F: FnMut(T, U) -> V,
{
    if lhs.shape() != rhs.shape() {
        return Err(AtlasNdError::InvalidArgument {
            op: "zip_map",
            reason: "array shapes must match",
        });
    }

    let data = value_iter(lhs.data(), lhs.offset(), lhs.shape(), lhs.strides())
        .copied()
        .zip(value_iter(rhs.data(), rhs.offset(), rhs.shape(), rhs.strides()).copied())
        .map(|(left, right)| op(left, right))
        .collect();

    Ok(NDArray::from_row_major_parts(lhs.shape().to_vec(), data)
        .expect("zip mapping preserves ndarray invariants"))
}
