use atlas_ndarray::{AtlasNdError, NDArray};

#[test]
fn axis_reductions_use_contiguous_lanes_for_negative_axes_and_keepdims() {
    let values = NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    assert_eq!(values.sum_axis(-1).unwrap().data(), &[6.0, 15.0]);
    assert_eq!(values.prod_axis(-1).unwrap().data(), &[6.0, 120.0]);
    assert_eq!(values.min_axis(-1).unwrap().data(), &[1.0, 4.0]);
    assert_eq!(values.max_axis(-1).unwrap().data(), &[3.0, 6.0]);
    assert_eq!(values.mean_axis(-1).unwrap().data(), &[2.0, 5.0]);

    let sum = values.sum_axis_keepdims(-1).unwrap();
    let mean = values.mean_axis_keepdims(-1).unwrap();
    assert_eq!(sum.shape(), &[2, 1]);
    assert_eq!(sum.data(), &[6.0, 15.0]);
    assert_eq!(mean.shape(), &[2, 1]);
    assert_eq!(mean.data(), &[2.0, 5.0]);
}

#[test]
fn truth_axis_reductions_use_contiguous_lanes() {
    let values =
        NDArray::from_shape_vec([2, 3], vec![true, true, false, true, true, true]).unwrap();

    assert_eq!(values.all_axis(-1).unwrap().data(), &[false, true]);
    assert_eq!(values.any_axis(-1).unwrap().data(), &[true, true]);

    let all = values.all_axis_keepdims(-1).unwrap();
    let any = values.any_axis_keepdims(-1).unwrap();
    assert_eq!(all.shape(), &[2, 1]);
    assert_eq!(all.data(), &[false, true]);
    assert_eq!(any.shape(), &[2, 1]);
    assert_eq!(any.data(), &[true, true]);
}

#[test]
fn axis_reductions_reject_axes_for_scalars() {
    let scalar = NDArray::from_shape_vec([], vec![1.0_f64]).unwrap();

    assert_eq!(scalar.sum_axis(0).unwrap_err(), AtlasNdError::InvalidAxis { axis: 0, ndim: 0 });
    assert_eq!(scalar.mean_axis(-1).unwrap_err(), AtlasNdError::InvalidAxis { axis: -1, ndim: 0 });
}

#[test]
fn axis_reductions_preserve_empty_lane_contracts() {
    let numeric = NDArray::<f64>::zeros([2, 0]).unwrap();
    let boolean = NDArray::<bool>::full([2, 0], false).unwrap();

    assert_eq!(numeric.sum_axis(-1).unwrap_err(), AtlasNdError::EmptyReduction { op: "sum" });
    assert_eq!(numeric.prod_axis(-1).unwrap_err(), AtlasNdError::EmptyReduction { op: "prod" });
    assert_eq!(numeric.min_axis(-1).unwrap_err(), AtlasNdError::EmptyReduction { op: "min" });
    assert_eq!(numeric.max_axis(-1).unwrap_err(), AtlasNdError::EmptyReduction { op: "max" });
    assert_eq!(numeric.mean_axis(-1).unwrap_err(), AtlasNdError::EmptyReduction { op: "mean" });
    assert_eq!(boolean.all_axis(-1).unwrap().data(), &[true, true]);
    assert_eq!(boolean.any_axis(-1).unwrap().data(), &[false, false]);
}
