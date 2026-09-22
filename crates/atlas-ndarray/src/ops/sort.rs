use std::cmp::Ordering;

use crate::{
    ArrayElement, AtlasNdResult, AxisIndex, NDArray, OperandMetadata, core::axis::normalize_axis,
    internal::value_iter, layout::element_count, view::ArrayView,
};

pub trait SortElement: ArrayElement {
    fn sort_compare(&self, other: &Self) -> Ordering;
}

macro_rules! impl_sort_element {
    ($($ty:ty),+ $(,)?) => { $(
        impl SortElement for $ty {
            fn sort_compare(&self, other: &Self) -> Ordering {
                self.cmp(other)
            }
        }
    )+ };
}

impl_sort_element!(bool, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

macro_rules! impl_float_sort_element {
    ($($ty:ty),+ $(,)?) => { $(
        impl SortElement for $ty {
            fn sort_compare(&self, other: &Self) -> Ordering {
                match (self.is_nan(), other.is_nan()) {
                    (true, true) => Ordering::Equal,
                    (true, false) => Ordering::Greater,
                    (false, true) => Ordering::Less,
                    (false, false) => self.total_cmp(other),
                }
            }
        }
    )+ };
}

impl_float_sort_element!(f32, f64);

impl<T: SortElement> NDArray<T> {
    /// Returns an owned array with each lane along `axis` sorted in ascending order.
    pub fn sort<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<Self> {
        sort_operand(self, axis)
    }
}

impl<'a, T: SortElement> ArrayView<'a, T> {
    /// Returns an owned array with each lane along `axis` sorted in ascending order.
    pub fn sort<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<T>> {
        sort_operand(self, axis)
    }
}

fn sort_operand<T, O, A>(operand: &O, axis: A) -> AtlasNdResult<NDArray<T>>
where
    T: SortElement,
    O: OperandMetadata<T> + ?Sized,
    A: AxisIndex,
{
    let axis = normalize_axis(axis, operand.ndim())?;
    let shape = operand.shape();
    let axis_len = shape[axis];
    let inner = element_count(&shape[axis + 1..]);
    let outer = element_count(&shape[..axis]);
    let data: Vec<_> =
        value_iter(operand.data(), operand.offset(), shape, operand.strides()).copied().collect();
    let mut sorted = data.clone();

    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let mut lane: Vec<_> = (0..axis_len)
                .map(|axis_index| data[(outer_index * axis_len + axis_index) * inner + inner_index])
                .collect();
            lane.sort_by(SortElement::sort_compare);
            for (axis_index, value) in lane.into_iter().enumerate() {
                sorted[(outer_index * axis_len + axis_index) * inner + inner_index] = value;
            }
        }
    }

    NDArray::from_shape_vec(shape.to_vec(), sorted)
}
