use atlas_ndarray::array::NDArray;

#[test]
fn view_iteration_preserves_logical_order_for_contiguous_layouts() {
    let array = NDArray::from_vec(vec![2, 2], vec![1_i32, 2, 3, 4]).unwrap();
    let values: Vec<_> = array.view().iter().copied().collect();

    assert_eq!(values, vec![1, 2, 3, 4]);
}

#[test]
fn view_iteration_preserves_logical_order_for_strided_layouts() {
    let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let values: Vec<_> = array.view().transpose().iter().copied().collect();

    assert_eq!(values, vec![0, 3, 1, 4, 2, 5]);
}

#[test]
fn sliced_view_iteration_uses_strided_traversal() {
    let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let slice = array.view().slice(&[0, 1], vec![2, 2]).unwrap();
    let values: Vec<_> = slice.iter().copied().collect();

    assert_eq!(values, vec![1, 2, 4, 5]);
}
