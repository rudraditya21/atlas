use atlas_ndarray::NDArray;
use proptest::prelude::*;

fn contiguous_slice_case() -> impl Strategy<Value = (Vec<i32>, usize, usize, usize, usize)> {
    (1_usize..=4, 1_usize..=4).prop_flat_map(|(rows, columns)| {
        (prop::collection::vec(any::<i32>(), rows * columns), 0..rows, 1..=rows).prop_filter_map(
            "slice length must fit",
            move |(data, start, length)| {
                (length <= rows - start).then_some((data, rows, columns, start, length))
            },
        )
    })
}

#[test]
fn reshaped_slices_iterate_logically_instead_of_exposing_backing_storage() {
    let array =
        NDArray::from_shape_vec([2, 2], vec![0_i32, 30_756_846, 1_719_564_484, 743_516_376])
            .unwrap();
    let reshaped = array.view().slice([1, 0], [1, 2]).unwrap().reshape([2]).unwrap();

    assert_eq!(reshaped.iter().copied().collect::<Vec<_>>(), &[1_719_564_484, 743_516_376]);
}

proptest! {
    #[test]
    fn slice_transpose_reshape_and_materialization_match_row_major_references(
        (data, rows, columns, start, length) in contiguous_slice_case(),
    ) {
        let array = NDArray::from_shape_vec([rows, columns], data.clone()).unwrap();
        let sliced = array.view().slice([start, 0], [length, columns]).unwrap();
        let expected_slice = data[start * columns..(start + length) * columns].to_vec();
        let reshaped = sliced.clone().reshape([length * columns]).unwrap();
        let transposed = sliced.transpose();
        let source = &data;
        let expected_transposed: Vec<_> = (0..columns)
            .flat_map(|column| {
                (0..length).map(move |row| source[(start + row) * columns + column])
            })
            .collect();
        let materialized = transposed.to_owned();
        let incremented = materialized.add(1_i32);
        let expected_incremented: Vec<_> = expected_transposed.iter().map(|value| value.wrapping_add(1)).collect();

        prop_assert_eq!(reshaped.iter().copied().collect::<Vec<_>>(), expected_slice);
        prop_assert_eq!(materialized.shape(), &[columns, length]);
        prop_assert_eq!(materialized.data(), expected_transposed.as_slice());
        prop_assert_eq!(incremented.data(), expected_incremented.as_slice());
    }
}
