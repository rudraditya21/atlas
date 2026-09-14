use atlas_ndarray::{AtlasNdError, NDArray};

#[test]
fn arg_reductions_keepdims_preserve_ties_and_first_nan_indices() {
    let values =
        NDArray::from_shape_vec([2, 3], vec![3.0_f64, 1.0, 1.0, f64::NAN, 2.0, f64::NAN]).unwrap();

    let argmin = values.argmin_axis_keepdims(1).unwrap();
    let argmax = values.argmax_axis_keepdims(1).unwrap();

    assert_eq!(argmin.shape(), &[2, 1]);
    assert_eq!(argmin.data(), &[1, 0]);
    assert_eq!(argmax.shape(), &[2, 1]);
    assert_eq!(argmax.data(), &[0, 0]);
}

#[test]
fn arg_reductions_keepdims_follow_logical_view_order_and_negative_axes() {
    let values = NDArray::from_shape_vec([2, 3], vec![3_i32, 1, 2, 6, 5, 4]).unwrap();
    let view = values.view().transpose();

    let argmin = view.argmin_axis_keepdims(-1).unwrap();
    let argmax = view.argmax_axis_keepdims(-1).unwrap();

    assert_eq!(argmin.shape(), &[3, 1]);
    assert_eq!(argmin.data(), &[0, 0, 0]);
    assert_eq!(argmax.shape(), &[3, 1]);
    assert_eq!(argmax.data(), &[1, 1, 1]);
}

#[test]
fn arg_reductions_keepdims_define_empty_lane_behavior() {
    let empty_lanes = NDArray::<i32>::zeros([0, 3]).unwrap();
    let empty_output = NDArray::<i32>::zeros([2, 0]).unwrap();

    assert_eq!(
        empty_lanes.argmin_axis_keepdims(0).unwrap_err(),
        AtlasNdError::EmptyReduction { op: "argmin" }
    );
    assert_eq!(
        empty_lanes.argmax_axis_keepdims(0).unwrap_err(),
        AtlasNdError::EmptyReduction { op: "argmax" }
    );
    assert_eq!(empty_output.argmin_axis_keepdims(0).unwrap().shape(), &[1, 0]);
    assert_eq!(empty_output.argmax_axis_keepdims(0).unwrap().shape(), &[1, 0]);
}
