use atlas_ndarray::NDArray;

#[test]
fn count_true_counts_contiguous_masks() {
    let mask =
        NDArray::from_shape_vec([2, 3], vec![true, false, true, false, true, false]).unwrap();

    assert_eq!(mask.count_true(), 3);
}

#[test]
fn count_true_supports_transposed_and_sliced_views() {
    let mask =
        NDArray::from_shape_vec([2, 3], vec![true, false, true, false, true, false]).unwrap();
    let transposed = mask.view().transpose();
    let sliced = mask.view().slice([0, 1], [2, 2]).unwrap();

    assert_eq!(transposed.count_true(), 3);
    assert_eq!(sliced.count_true(), 2);
}

#[test]
fn count_true_handles_scalar_and_empty_masks() {
    let scalar = NDArray::from_shape_vec([], vec![true]).unwrap();
    let empty = NDArray::from_shape_vec([2, 0, 3], Vec::<bool>::new()).unwrap();

    assert_eq!(scalar.count_true(), 1);
    assert_eq!(empty.view().count_true(), 0);
}
