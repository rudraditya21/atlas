use atlas_ndarray::NDArray;

#[test]
fn nonzero_indices_returns_contiguous_logical_coordinates() {
    let values = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 0, 2, 3, 0]).unwrap();

    let indices = values.nonzero_indices().unwrap();

    assert_eq!(indices.shape(), &[3, 2]);
    assert_eq!(indices.data(), &[0, 1, 1, 0, 1, 1]);
    assert!(indices.is_contiguous());
}

#[test]
fn nonzero_indices_uses_transposed_and_sliced_view_coordinates() {
    let values = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 0, 2, 3, 0]).unwrap();
    let transposed = values.view().transpose().nonzero_indices().unwrap();
    let sliced = values.view().slice([0, 1], [2, 2]).unwrap().nonzero_indices().unwrap();

    assert_eq!(transposed.shape(), &[3, 2]);
    assert_eq!(transposed.data(), &[0, 1, 1, 0, 1, 1]);
    assert_eq!(sliced.shape(), &[2, 2]);
    assert_eq!(sliced.data(), &[0, 0, 1, 0]);
}

#[test]
fn nonzero_indices_handles_scalar_and_empty_inputs() {
    let scalar = NDArray::from_shape_vec([], vec![5_i32]).unwrap();
    let zero_scalar = NDArray::from_shape_vec([], vec![0_i32]).unwrap();
    let empty = NDArray::<i32>::zeros([2, 0, 3]).unwrap();

    assert_eq!(scalar.nonzero_indices().unwrap().shape(), &[1, 0]);
    assert_eq!(zero_scalar.nonzero_indices().unwrap().shape(), &[0, 0]);
    assert_eq!(empty.view().nonzero_indices().unwrap().shape(), &[0, 3]);
}

#[test]
fn nonzero_indices_supports_boolean_inputs() {
    let values = NDArray::from_shape_vec([2, 2], vec![false, true, true, false]).unwrap();

    let indices = values.nonzero_indices().unwrap();

    assert_eq!(indices.shape(), &[2, 2]);
    assert_eq!(indices.data(), &[0, 1, 1, 0]);
}
