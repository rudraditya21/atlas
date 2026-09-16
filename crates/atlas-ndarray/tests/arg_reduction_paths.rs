use atlas_ndarray::NDArray;

#[test]
fn arg_axis_reductions_preserve_first_ties_and_keepdims() {
    let values = NDArray::from_shape_vec([2, 3], vec![2.0_f64, 1.0, 1.0, 4.0, 4.0, 5.0]).unwrap();

    assert_eq!(values.argmin_axis(-1).unwrap().data(), &[1, 0]);
    assert_eq!(values.argmax_axis(-1).unwrap().data(), &[0, 2]);

    let min = values.argmin_axis_keepdims(-1).unwrap();
    let max = values.argmax_axis_keepdims(-1).unwrap();
    assert_eq!(min.shape(), &[2, 1]);
    assert_eq!(min.data(), &[1, 0]);
    assert_eq!(max.shape(), &[2, 1]);
    assert_eq!(max.data(), &[0, 2]);
}

#[test]
fn arg_axis_reductions_preserve_first_nan_for_dense_and_strided_layouts() {
    let values =
        NDArray::from_shape_vec([2, 3], vec![1.0_f64, f64::NAN, 3.0, 4.0, 5.0, f64::NAN]).unwrap();

    assert_eq!(values.argmin_axis(-1).unwrap().data(), &[1, 2]);
    assert_eq!(values.argmax_axis(-1).unwrap().data(), &[1, 2]);

    let transposed = values.view().transpose();
    assert_eq!(transposed.argmin_axis(0).unwrap().data(), &[1, 2]);
    assert_eq!(transposed.argmax_axis(0).unwrap().data(), &[1, 2]);
}
