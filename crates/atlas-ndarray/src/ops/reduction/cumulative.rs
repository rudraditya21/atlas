use crate::{
    AtlasNdResult, AxisIndex, ElementwiseArithmetic, NDArray, core::axis::normalize_axis,
    internal::value_iter, layout::element_count,
};

pub(super) fn cumsum<T: ElementwiseArithmetic>(
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> AtlasNdResult<NDArray<T>> {
    cumulative(
        data,
        offset,
        shape,
        strides,
        None,
        T::zero(),
        ElementwiseArithmetic::elementwise_add,
    )
}

pub(super) fn cumprod<T: ElementwiseArithmetic>(
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> AtlasNdResult<NDArray<T>> {
    cumulative(data, offset, shape, strides, None, T::one(), ElementwiseArithmetic::elementwise_mul)
}

pub(super) fn cumsum_axis<T: ElementwiseArithmetic>(
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<T>> {
    cumulative(
        data,
        offset,
        shape,
        strides,
        Some(normalize_axis(axis, shape.len())?),
        T::zero(),
        ElementwiseArithmetic::elementwise_add,
    )
}

pub(super) fn cumprod_axis<T: ElementwiseArithmetic>(
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<T>> {
    cumulative(
        data,
        offset,
        shape,
        strides,
        Some(normalize_axis(axis, shape.len())?),
        T::one(),
        ElementwiseArithmetic::elementwise_mul,
    )
}

fn cumulative<T: ElementwiseArithmetic>(
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: Option<usize>,
    initial: T,
    operation: impl Fn(T, T) -> T,
) -> AtlasNdResult<NDArray<T>> {
    let mut values: Vec<T> = value_iter(data, offset, shape, strides).copied().collect();

    if let Some(axis) = axis {
        let outer = element_count(&shape[..axis]);
        let axis_len = shape[axis];
        let inner = element_count(&shape[axis + 1..]);

        for outer_index in 0..outer {
            for inner_index in 0..inner {
                let mut total = initial;
                for axis_index in 0..axis_len {
                    let index = (outer_index * axis_len + axis_index) * inner + inner_index;
                    total = operation(total, values[index]);
                    values[index] = total;
                }
            }
        }
    } else {
        let mut total = initial;
        for value in &mut values {
            total = operation(total, *value);
            *value = total;
        }
    }

    NDArray::from_shape_vec(shape.to_vec(), values)
}
