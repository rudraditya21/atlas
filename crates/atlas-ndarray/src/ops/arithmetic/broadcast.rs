use super::from_owned_parts;
use crate::{
    NDArray, Numeric,
    internal::{broadcast_offset_pair_iter, shape::compute_strides, simd},
    layout::{broadcast::BroadcastMetadata, element_count},
};

pub(super) fn elementwise_binary_broadcast<T, F>(
    lhs: &NDArray<T>,
    rhs: &NDArray<T>,
    metadata: BroadcastMetadata,
    op: F,
) -> NDArray<T>
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
{
    if let Some(result) = try_elementwise_binary_broadcast_fast_path(lhs, rhs, &metadata, op) {
        return result;
    }

    let output_len = element_count(&metadata.shape);
    let mut data = vec![T::zero(); output_len];

    for (slot, (lhs_offset, rhs_offset)) in data.iter_mut().zip(broadcast_offset_pair_iter(
        0,
        0,
        &metadata.shape,
        &metadata.lhs_strides,
        &metadata.rhs_strides,
    )) {
        *slot = op(lhs.data()[lhs_offset], rhs.data()[rhs_offset]);
    }

    from_owned_parts(metadata.shape, data)
}

fn try_elementwise_binary_broadcast_fast_path<T, F>(
    lhs: &NDArray<T>,
    rhs: &NDArray<T>,
    metadata: &BroadcastMetadata,
    op: F,
) -> Option<NDArray<T>>
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
{
    if is_output_contiguous(&metadata.shape, &metadata.lhs_strides)
        && is_scalar_broadcast(&metadata.rhs_strides, rhs.data().len())
    {
        return Some(elementwise_scalar_broadcast_rhs(
            lhs.data(),
            rhs.data()[0],
            &metadata.shape,
            op,
        ));
    }

    if is_output_contiguous(&metadata.shape, &metadata.rhs_strides)
        && is_scalar_broadcast(&metadata.lhs_strides, lhs.data().len())
    {
        return Some(elementwise_scalar_broadcast_lhs(
            lhs.data()[0],
            rhs.data(),
            &metadata.shape,
            op,
        ));
    }

    if let Some(result) = try_elementwise_row_broadcast(lhs, rhs, metadata, op) {
        return Some(result);
    }

    try_elementwise_column_broadcast(lhs, rhs, metadata, op)
}

fn try_elementwise_row_broadcast<T, F>(
    lhs: &NDArray<T>,
    rhs: &NDArray<T>,
    metadata: &BroadcastMetadata,
    op: F,
) -> Option<NDArray<T>>
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
{
    let [rows, cols] = metadata.shape.as_slice() else {
        return None;
    };

    if is_output_contiguous(&metadata.shape, &metadata.lhs_strides)
        && is_row_broadcast_strides(&metadata.rhs_strides)
        && rhs.data().len() == *cols
    {
        return Some(elementwise_row_broadcast_rhs(lhs.data(), rhs.data(), *rows, *cols, op));
    }

    if is_output_contiguous(&metadata.shape, &metadata.rhs_strides)
        && is_row_broadcast_strides(&metadata.lhs_strides)
        && lhs.data().len() == *cols
    {
        return Some(elementwise_row_broadcast_lhs(lhs.data(), rhs.data(), *rows, *cols, op));
    }

    None
}

fn try_elementwise_column_broadcast<T, F>(
    lhs: &NDArray<T>,
    rhs: &NDArray<T>,
    metadata: &BroadcastMetadata,
    op: F,
) -> Option<NDArray<T>>
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
{
    let [rows, cols] = metadata.shape.as_slice() else {
        return None;
    };

    if is_output_contiguous(&metadata.shape, &metadata.lhs_strides)
        && is_column_broadcast_strides(&metadata.rhs_strides)
        && rhs.data().len() == *rows
    {
        return Some(elementwise_column_broadcast_rhs(lhs.data(), rhs.data(), *rows, *cols, op));
    }

    if is_output_contiguous(&metadata.shape, &metadata.rhs_strides)
        && is_column_broadcast_strides(&metadata.lhs_strides)
        && lhs.data().len() == *rows
    {
        return Some(elementwise_column_broadcast_lhs(lhs.data(), rhs.data(), *rows, *cols, op));
    }

    None
}

fn elementwise_scalar_broadcast_rhs<T, F>(
    input: &[T],
    scalar: T,
    shape: &[usize],
    op: F,
) -> NDArray<T>
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
{
    let mut data = vec![T::zero(); input.len()];
    simd::map_scalar_contiguous(input, scalar, &mut data, op);

    from_owned_parts(shape.to_vec(), data)
}

fn elementwise_scalar_broadcast_lhs<T, F>(
    scalar: T,
    input: &[T],
    shape: &[usize],
    op: F,
) -> NDArray<T>
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
{
    let mut data = vec![T::zero(); input.len()];
    simd::map_scalar_contiguous(input, scalar, &mut data, |value, rhs| op(rhs, value));

    from_owned_parts(shape.to_vec(), data)
}

fn elementwise_row_broadcast_rhs<T, F>(
    matrix: &[T],
    row: &[T],
    rows: usize,
    cols: usize,
    op: F,
) -> NDArray<T>
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
{
    let mut data = vec![T::zero(); rows * cols];

    for (matrix_row, out_row) in matrix.chunks_exact(cols).zip(data.chunks_exact_mut(cols)) {
        simd::map_binary_contiguous(matrix_row, row, out_row, op);
    }

    from_owned_parts(vec![rows, cols], data)
}

fn elementwise_row_broadcast_lhs<T, F>(
    row: &[T],
    matrix: &[T],
    rows: usize,
    cols: usize,
    op: F,
) -> NDArray<T>
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
{
    let mut data = vec![T::zero(); rows * cols];

    for (matrix_row, out_row) in matrix.chunks_exact(cols).zip(data.chunks_exact_mut(cols)) {
        simd::map_binary_contiguous(row, matrix_row, out_row, op);
    }

    from_owned_parts(vec![rows, cols], data)
}

fn elementwise_column_broadcast_rhs<T, F>(
    matrix: &[T],
    column: &[T],
    rows: usize,
    cols: usize,
    op: F,
) -> NDArray<T>
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
{
    let mut data = vec![T::zero(); rows * cols];

    for ((matrix_row, out_row), &scalar) in
        matrix.chunks_exact(cols).zip(data.chunks_exact_mut(cols)).zip(column.iter())
    {
        simd::map_scalar_contiguous(matrix_row, scalar, out_row, op);
    }

    from_owned_parts(vec![rows, cols], data)
}

fn elementwise_column_broadcast_lhs<T, F>(
    column: &[T],
    matrix: &[T],
    rows: usize,
    cols: usize,
    op: F,
) -> NDArray<T>
where
    T: Numeric,
    F: Fn(T, T) -> T + Copy,
{
    let mut data = vec![T::zero(); rows * cols];

    for ((matrix_row, out_row), &scalar) in
        matrix.chunks_exact(cols).zip(data.chunks_exact_mut(cols)).zip(column.iter())
    {
        simd::map_scalar_contiguous(matrix_row, scalar, out_row, |value, rhs| op(rhs, value));
    }

    from_owned_parts(vec![rows, cols], data)
}

fn is_output_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    strides == compute_strides(shape)
}

fn is_scalar_broadcast(strides: &[usize], len: usize) -> bool {
    len == 1 && strides.iter().all(|&stride| stride == 0)
}

fn is_row_broadcast_strides(strides: &[usize]) -> bool {
    matches!(strides, [0, 1])
}

fn is_column_broadcast_strides(strides: &[usize]) -> bool {
    matches!(strides, [1, 0])
}
