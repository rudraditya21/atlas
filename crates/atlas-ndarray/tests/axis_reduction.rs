use atlas_ndarray::{AtlasNdError, NDArray};

#[test]
fn axis_reductions_return_predictable_shapes() {
    let array = NDArray::from_shape_vec(vec![2, 3, 4], (0_i32..24).collect()).unwrap();

    assert_eq!(array.sum_axis(0).unwrap().shape(), &[3, 4]);
    assert_eq!(array.sum_axis(1).unwrap().shape(), &[2, 4]);
    assert_eq!(array.sum_axis(2).unwrap().shape(), &[2, 3]);
}

#[test]
fn axis_reductions_support_scalar_outputs() {
    let array = NDArray::from_shape_vec(vec![4], vec![1_i32, 2, 3, 4]).unwrap();

    assert_eq!(array.sum_axis(0).unwrap().shape(), &[] as &[usize]);
    assert_eq!(array.sum_axis(0).unwrap().data(), &[10]);
    assert_eq!(array.mean_axis(0).unwrap().data(), &[2.5]);
}

#[test]
fn axis_reductions_work_for_sliced_views() {
    let array = NDArray::from_shape_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let view = array.view().slice(&[0, 1], vec![2, 2]).unwrap();

    assert_eq!(view.sum_axis(0).unwrap().data(), &[5, 7]);
    assert_eq!(view.sum_axis(1).unwrap().data(), &[3, 9]);
}

#[test]
fn axis_reductions_report_invalid_axis_consistently() {
    let array = NDArray::new(vec![2, 2], 1_i32);

    assert_eq!(
        array.mean_axis(3).unwrap_err(),
        AtlasNdError::InvalidAxis { axis: 3, ndim: 2 }
    );
}
