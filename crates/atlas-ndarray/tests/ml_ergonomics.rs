use atlas_ndarray::NDArray;

#[test]
fn constructors_accept_array_and_slice_shapes() {
    let dynamic_shape = vec![2, 3];

    let zeros = NDArray::<i32>::zeros([2, 3]);
    let ones = NDArray::<i32>::ones([2, 3]);
    let full = NDArray::full(dynamic_shape.as_slice(), 7_i32);

    assert_eq!(zeros.shape(), &[2, 3]);
    assert_eq!(ones.shape(), &[2, 3]);
    assert_eq!(full.shape(), &[2, 3]);
}

#[test]
fn view_ops_accept_array_style_shapes() {
    let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let reshaped = array.view().reshape([3, 2]).unwrap();
    let sliced = array.view().slice([0, 1], [2, 2]).unwrap();

    assert_eq!(reshaped.shape(), &[3, 2]);
    assert_eq!(sliced.shape(), &[2, 2]);
}

#[test]
fn axis_reductions_support_negative_axes() {
    let array = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();

    assert_eq!(array.sum_axis(-1).unwrap().data(), &[6, 15]);
    assert_eq!(array.sum_axis(-2).unwrap().data(), &[5, 7, 9]);
}
