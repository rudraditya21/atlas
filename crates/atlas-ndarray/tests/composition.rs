use atlas_ndarray::NDArray;
use proptest::prelude::*;

const MAX_RANK: usize = 4;
const MAX_DIMENSION: usize = 4;

fn non_empty_array_strategy() -> impl Strategy<Value = (NDArray<i32>, usize, usize)> {
    prop::collection::vec(1..=MAX_DIMENSION, 1..=MAX_RANK).prop_flat_map(|shape| {
        let len: usize = shape.iter().product();
        (prop::collection::vec(any::<i32>(), len), 0..shape.len(), 1_usize..=6).prop_map(
            move |(data, axis, sections)| {
                (
                    NDArray::from_shape_vec(shape.clone(), data)
                        .expect("generated shape matches data"),
                    axis,
                    sections,
                )
            },
        )
    })
}

fn stack_case_strategy() -> impl Strategy<Value = (NDArray<i32>, NDArray<i32>, usize)> {
    prop::collection::vec(1..=MAX_DIMENSION, 1..=MAX_RANK).prop_flat_map(|shape| {
        let len: usize = shape.iter().product();
        (
            prop::collection::vec(any::<i32>(), len),
            prop::collection::vec(any::<i32>(), len),
            0..=shape.len(),
        )
            .prop_map(move |(left, right, axis)| {
                (
                    NDArray::from_shape_vec(shape.clone(), left)
                        .expect("generated shape matches left data"),
                    NDArray::from_shape_vec(shape.clone(), right)
                        .expect("generated shape matches right data"),
                    axis,
                )
            })
    })
}

fn stacked_values(left: &[i32], right: &[i32], shape: &[usize], axis: usize) -> Vec<i32> {
    let outer = shape[..axis].iter().product::<usize>();
    let inner = shape[axis..].iter().product::<usize>();
    let mut values = Vec::with_capacity(left.len() + right.len());

    for outer_index in 0..outer {
        let range = outer_index * inner..(outer_index + 1) * inner;
        values.extend_from_slice(&left[range.clone()]);
        values.extend_from_slice(&right[range]);
    }

    values
}

proptest! {
    #[test]
    fn split_then_concatenate_reconstructs_logical_values(
        (array, axis, sections) in non_empty_array_strategy(),
    ) {
        let parts = array.array_split(sections, axis).unwrap();
        let reconstructed = NDArray::concatenate(&parts, axis).unwrap();

        prop_assert_eq!(reconstructed.shape(), array.shape());
        prop_assert_eq!(reconstructed.data(), array.data());
    }

    #[test]
    fn stack_orders_contiguous_inputs_by_insertion_axis(
        (left, right, axis) in stack_case_strategy(),
    ) {
        let expected_shape = {
            let mut shape = left.shape().to_vec();
            shape.insert(axis, 2);
            shape
        };
        let expected_values = stacked_values(left.data(), right.data(), left.shape(), axis);
        let stacked = NDArray::stack(&[left.view(), right.view()], axis).unwrap();

        prop_assert_eq!(stacked.shape(), expected_shape.as_slice());
        prop_assert_eq!(stacked.data(), expected_values.as_slice());
    }

    #[test]
    fn stack_orders_strided_inputs_by_logical_iteration(
        (left, right, axis) in stack_case_strategy(),
    ) {
        let left = left.view().transpose();
        let right = right.view().transpose();
        let left_values: Vec<_> = left.iter().copied().collect();
        let right_values: Vec<_> = right.iter().copied().collect();
        let expected_values = stacked_values(&left_values, &right_values, left.shape(), axis);
        let expected_shape = {
            let mut shape = left.shape().to_vec();
            shape.insert(axis, 2);
            shape
        };
        let stacked = NDArray::stack(&[left, right], axis).unwrap();

        prop_assert_eq!(stacked.shape(), expected_shape.as_slice());
        prop_assert_eq!(stacked.data(), expected_values.as_slice());
    }
}
