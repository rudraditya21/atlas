use crate::{
    ArrayElement, AtlasNdError, AtlasNdResult, NDArray, OperandMetadata,
    internal::{layout::is_contiguous_layout, value_iter},
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

    /// Maps pairs of logical values and their multidimensional indices into a contiguous array.
    pub fn zip_map_indexed<U, V, O, F>(&self, rhs: &O, op: F) -> AtlasNdResult<NDArray<V>>
    where
        U: ArrayElement,
        V: ArrayElement,
        O: OperandMetadata<U> + ?Sized,
        F: FnMut(&[usize], T, U) -> V,
    {
        zip_map_indexed_operands(self, rhs, op)
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

    /// Maps pairs of logical values and their multidimensional indices into a contiguous array.
    pub fn zip_map_indexed<U, V, O, F>(&self, rhs: &O, op: F) -> AtlasNdResult<NDArray<V>>
    where
        U: ArrayElement,
        V: ArrayElement,
        O: OperandMetadata<U> + ?Sized,
        F: FnMut(&[usize], T, U) -> V,
    {
        zip_map_indexed_operands(self, rhs, op)
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
    validate_matching_shapes(lhs, rhs, "zip_map")?;

    let data = if is_contiguous_layout(lhs.shape(), lhs.strides())
        && is_contiguous_layout(rhs.shape(), rhs.strides())
    {
        lhs.dense_slice()
            .expect("contiguous operands always expose dense storage slices")
            .iter()
            .copied()
            .zip(
                rhs.dense_slice()
                    .expect("contiguous operands always expose dense storage slices")
                    .iter()
                    .copied(),
            )
            .map(|(left, right)| op(left, right))
            .collect()
    } else {
        value_iter(lhs.data(), lhs.offset(), lhs.shape(), lhs.strides())
            .copied()
            .zip(value_iter(rhs.data(), rhs.offset(), rhs.shape(), rhs.strides()).copied())
            .map(|(left, right)| op(left, right))
            .collect()
    };

    Ok(NDArray::from_row_major_parts(lhs.shape().to_vec(), data)
        .expect("zip mapping preserves ndarray invariants"))
}

fn zip_map_indexed_operands<T, U, V, L, R, F>(
    lhs: &L,
    rhs: &R,
    mut op: F,
) -> AtlasNdResult<NDArray<V>>
where
    T: ArrayElement,
    U: ArrayElement,
    V: ArrayElement,
    L: OperandMetadata<T> + ?Sized,
    R: OperandMetadata<U> + ?Sized,
    F: FnMut(&[usize], T, U) -> V,
{
    validate_matching_shapes(lhs, rhs, "zip_map_indexed")?;

    let mut index = vec![0; lhs.shape().len()];
    let mut data = Vec::new();
    for (linear_index, (left, right)) in
        value_iter(lhs.data(), lhs.offset(), lhs.shape(), lhs.strides())
            .copied()
            .zip(value_iter(rhs.data(), rhs.offset(), rhs.shape(), rhs.strides()).copied())
            .enumerate()
    {
        logical_index(linear_index, lhs.shape(), &mut index);
        data.push(op(&index, left, right));
    }

    Ok(NDArray::from_row_major_parts(lhs.shape().to_vec(), data)
        .expect("indexed zip mapping preserves ndarray invariants"))
}

fn validate_matching_shapes<T, U, L, R>(lhs: &L, rhs: &R, op: &'static str) -> AtlasNdResult<()>
where
    T: ArrayElement,
    U: ArrayElement,
    L: OperandMetadata<T> + ?Sized,
    R: OperandMetadata<U> + ?Sized,
{
    if lhs.shape() != rhs.shape() {
        return Err(AtlasNdError::InvalidArgument { op, reason: "array shapes must match" });
    }

    Ok(())
}

fn logical_index(mut linear_index: usize, shape: &[usize], index: &mut [usize]) {
    for axis in (0..shape.len()).rev() {
        index[axis] = linear_index % shape[axis];
        linear_index /= shape[axis];
    }
}
