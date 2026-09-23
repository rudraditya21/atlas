use std::cmp::Ordering;

use crate::{
    ArrayElement, AtlasNdResult, AxisIndex, NDArray, OperandMetadata, core::axis::normalize_axis,
    internal::value_iter, layout::element_count, view::ArrayView,
};

pub trait SortElement: ArrayElement {
    fn sort_compare(&self, other: &Self) -> Ordering;

    fn sort_equal(&self, other: &Self) -> bool;
}

macro_rules! impl_sort_element {
    ($($ty:ty),+ $(,)?) => { $(
        impl SortElement for $ty {
            fn sort_compare(&self, other: &Self) -> Ordering {
                self.cmp(other)
            }

            fn sort_equal(&self, other: &Self) -> bool {
                self == other
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

            fn sort_equal(&self, other: &Self) -> bool {
                self == other || (self.is_nan() && other.is_nan())
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

    /// Returns stable ascending sort indices for each lane along `axis`.
    pub fn argsort<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<i64>> {
        argsort_operand(self, axis)
    }

    /// Returns indices that partition each lane along `axis` at `kth`.
    pub fn argpartition<A: AxisIndex>(&self, kth: usize, axis: A) -> AtlasNdResult<NDArray<i64>> {
        argpartition_operand(self, kth, axis)
    }

    /// Partitions each lane along `axis` so the value at `kth` is in sorted position.
    pub fn partition<A: AxisIndex>(&self, kth: usize, axis: A) -> AtlasNdResult<Self> {
        partition_operand(self, kth, axis)
    }

    /// Returns sorted unique logical values as an owned one-dimensional array.
    pub fn unique(&self) -> NDArray<T> {
        unique_operand(self)
    }
}

impl<'a, T: SortElement> ArrayView<'a, T> {
    /// Returns an owned array with each lane along `axis` sorted in ascending order.
    pub fn sort<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<T>> {
        sort_operand(self, axis)
    }

    /// Returns stable ascending sort indices for each lane along `axis`.
    pub fn argsort<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<i64>> {
        argsort_operand(self, axis)
    }

    /// Returns indices that partition each lane along `axis` at `kth`.
    pub fn argpartition<A: AxisIndex>(&self, kth: usize, axis: A) -> AtlasNdResult<NDArray<i64>> {
        argpartition_operand(self, kth, axis)
    }

    /// Partitions each lane along `axis` so the value at `kth` is in sorted position.
    pub fn partition<A: AxisIndex>(&self, kth: usize, axis: A) -> AtlasNdResult<NDArray<T>> {
        partition_operand(self, kth, axis)
    }

    /// Returns sorted unique logical values as an owned one-dimensional array.
    pub fn unique(&self) -> NDArray<T> {
        unique_operand(self)
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

fn argsort_operand<T, O, A>(operand: &O, axis: A) -> AtlasNdResult<NDArray<i64>>
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
    let mut indices = vec![0; data.len()];

    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let mut lane: Vec<_> = (0..axis_len).collect();
            lane.sort_by(|&left, &right| {
                data[(outer_index * axis_len + left) * inner + inner_index]
                    .sort_compare(&data[(outer_index * axis_len + right) * inner + inner_index])
            });
            for (axis_index, source_index) in lane.into_iter().enumerate() {
                indices[(outer_index * axis_len + axis_index) * inner + inner_index] =
                    i64::try_from(source_index).expect("ndarray axes fit i64");
            }
        }
    }

    NDArray::from_shape_vec(shape.to_vec(), indices)
}

fn partition_operand<T, O, A>(operand: &O, kth: usize, axis: A) -> AtlasNdResult<NDArray<T>>
where
    T: SortElement,
    O: OperandMetadata<T> + ?Sized,
    A: AxisIndex,
{
    let axis = normalize_axis(axis, operand.ndim())?;
    let shape = operand.shape();
    let axis_len = shape[axis];
    if kth >= axis_len {
        return Err(crate::AtlasNdError::IndexOutOfBounds {
            axis,
            index: i64::try_from(kth).expect("ndarray axes fit i64"),
            dim: axis_len,
        });
    }

    let inner = element_count(&shape[axis + 1..]);
    let outer = element_count(&shape[..axis]);
    let data: Vec<_> =
        value_iter(operand.data(), operand.offset(), shape, operand.strides()).copied().collect();
    let mut partitioned = data.clone();

    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let mut lane: Vec<_> = (0..axis_len)
                .map(|axis_index| data[(outer_index * axis_len + axis_index) * inner + inner_index])
                .collect();
            lane.select_nth_unstable_by(kth, SortElement::sort_compare);
            for (axis_index, value) in lane.into_iter().enumerate() {
                partitioned[(outer_index * axis_len + axis_index) * inner + inner_index] = value;
            }
        }
    }

    NDArray::from_shape_vec(shape.to_vec(), partitioned)
}

fn argpartition_operand<T, O, A>(operand: &O, kth: usize, axis: A) -> AtlasNdResult<NDArray<i64>>
where
    T: SortElement,
    O: OperandMetadata<T> + ?Sized,
    A: AxisIndex,
{
    let axis = normalize_axis(axis, operand.ndim())?;
    let shape = operand.shape();
    let axis_len = shape[axis];
    if kth >= axis_len {
        return Err(crate::AtlasNdError::IndexOutOfBounds {
            axis,
            index: i64::try_from(kth).expect("ndarray axes fit i64"),
            dim: axis_len,
        });
    }

    let inner = element_count(&shape[axis + 1..]);
    let outer = element_count(&shape[..axis]);
    let data: Vec<_> =
        value_iter(operand.data(), operand.offset(), shape, operand.strides()).copied().collect();
    let mut indices = vec![0; data.len()];

    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let mut lane: Vec<_> = (0..axis_len).collect();
            lane.select_nth_unstable_by(kth, |left, right| {
                data[(outer_index * axis_len + *left) * inner + inner_index]
                    .sort_compare(&data[(outer_index * axis_len + *right) * inner + inner_index])
            });
            for (axis_index, source_index) in lane.into_iter().enumerate() {
                indices[(outer_index * axis_len + axis_index) * inner + inner_index] =
                    i64::try_from(source_index).expect("ndarray axes fit i64");
            }
        }
    }

    NDArray::from_shape_vec(shape.to_vec(), indices)
}

fn unique_operand<T, O>(operand: &O) -> NDArray<T>
where
    T: SortElement,
    O: OperandMetadata<T> + ?Sized,
{
    let mut values: Vec<_> =
        value_iter(operand.data(), operand.offset(), operand.shape(), operand.strides())
            .copied()
            .collect();
    values.sort_by(SortElement::sort_compare);
    values.dedup_by(|left, right| left.sort_equal(right));

    NDArray::from_vector_data(values).expect("unique preserves ndarray invariants")
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::SortElement;
    use crate::{AtlasNdError, NDArray};

    #[test]
    fn partition_places_the_requested_owned_axis_value_and_preserves_values() {
        let values = NDArray::from_shape_vec([5], vec![9_i32, 1, 8, 2, 7]).unwrap();
        let partitioned = values.partition(2, -1).unwrap();

        assert_eq!(partitioned.data()[2], 7);
        assert!(partitioned.data()[..2].iter().all(|value| *value <= partitioned.data()[2]));
        assert!(partitioned.data()[3..].iter().all(|value| *value >= partitioned.data()[2]));

        let mut actual = partitioned.data().to_vec();
        let mut expected = values.data().to_vec();
        actual.sort();
        expected.sort();
        assert_eq!(actual, expected);
    }

    #[test]
    fn partition_uses_logical_values_from_non_contiguous_views() {
        let values = NDArray::from_shape_vec([2, 3], vec![9_i32, 1, 8, 2, 7, 3]).unwrap();
        let view = values.view().transpose();
        let partitioned = view.partition(1, 0).unwrap();

        assert_eq!(partitioned.shape(), &[3, 2]);
        assert_eq!(&partitioned.data()[2..4], &[8, 3]);
        for column in 0..2 {
            let pivot = partitioned.data()[2 + column];
            assert!(partitioned.data()[column] <= pivot);
            assert!(partitioned.data()[4 + column] >= pivot);
        }

        let mut actual = partitioned.data().to_vec();
        let mut expected: Vec<_> = view.iter().copied().collect();
        actual.sort();
        expected.sort();
        assert_eq!(actual, expected);
    }

    #[test]
    fn partition_rejects_an_out_of_bounds_kth_value() {
        let values = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();

        assert_eq!(
            values.partition(3, 1).unwrap_err(),
            AtlasNdError::IndexOutOfBounds { axis: 1, index: 3, dim: 3 }
        );
    }

    #[test]
    fn partition_handles_duplicate_values() {
        let values = NDArray::from_shape_vec([6], vec![3_i32, 1, 2, 2, 2, 4]).unwrap();
        let partitioned = values.partition(3, 0).unwrap();

        assert_eq!(partitioned.data()[3], 2);
        assert!(partitioned.data()[..3].iter().all(|value| *value <= partitioned.data()[3]));
        assert!(partitioned.data()[4..].iter().all(|value| *value >= partitioned.data()[3]));
    }

    #[test]
    fn partition_orders_nan_values_using_sort_element_ordering() {
        let values = NDArray::from_shape_vec([5], vec![f64::NAN, 2.0, 1.0, f64::NAN, 0.0]).unwrap();
        let partitioned = values.partition(3, 0).unwrap();
        let pivot = partitioned.data()[3];

        assert!(pivot.is_nan());
        assert!(
            partitioned.data()[..3]
                .iter()
                .all(|value| value.sort_compare(&pivot) != Ordering::Greater)
        );
        assert!(
            partitioned.data()[4..]
                .iter()
                .all(|value| value.sort_compare(&pivot) != Ordering::Less)
        );
        assert_eq!(partitioned.data().iter().filter(|value| value.is_nan()).count(), 2);
    }

    #[test]
    fn partition_rejects_zero_length_lanes_and_scalar_axes() {
        let empty = NDArray::<i32>::zeros([2, 0, 3]).unwrap();
        let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();

        assert_eq!(
            empty.partition(0, 1).unwrap_err(),
            AtlasNdError::IndexOutOfBounds { axis: 1, index: 0, dim: 0 }
        );
        assert_eq!(
            scalar.partition(0, -1).unwrap_err(),
            AtlasNdError::InvalidAxis { axis: -1, ndim: 0 }
        );
    }

    #[test]
    fn argpartition_returns_a_partitioned_permutation() {
        let values = NDArray::from_shape_vec([5], vec![9_i32, 1, 8, 2, 7]).unwrap();
        let indices = values.argpartition(2, -1).unwrap();
        let partitioned: Vec<_> =
            indices.data().iter().map(|&index| values.data()[index as usize]).collect();

        assert_eq!(partitioned[2], 7);
        assert!(partitioned[..2].iter().all(|value| *value <= partitioned[2]));
        assert!(partitioned[3..].iter().all(|value| *value >= partitioned[2]));

        let mut actual = indices.data().to_vec();
        actual.sort();
        assert_eq!(actual, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn argpartition_uses_logical_view_values_and_nan_ordering() {
        let values = NDArray::from_shape_vec([2, 3], vec![9_i32, 1, 8, 2, 7, 3]).unwrap();
        let view = values.view().transpose();
        let indices = view.argpartition(1, 0).unwrap();
        let expected = [8, 3];

        for column in 0..2 {
            let lane: Vec<_> = (0..3)
                .map(|row| *view.get(&[indices.data()[row * 2 + column], column as i64]).unwrap())
                .collect();
            assert_eq!(lane[1], expected[column]);
            assert!(lane[0] <= lane[1]);
            assert!(lane[2] >= lane[1]);
        }

        let values = NDArray::from_shape_vec([5], vec![f64::NAN, 2.0, 1.0, f64::NAN, 0.0]).unwrap();
        let indices = values.argpartition(3, 0).unwrap();
        let partitioned: Vec<_> =
            indices.data().iter().map(|&index| values.data()[index as usize]).collect();
        let pivot = partitioned[3];

        assert!(pivot.is_nan());
        assert!(
            partitioned[..3].iter().all(|value| value.sort_compare(&pivot) != Ordering::Greater)
        );
        assert!(partitioned[4..].iter().all(|value| value.sort_compare(&pivot) != Ordering::Less));
    }
}
