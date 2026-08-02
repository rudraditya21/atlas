use atlas_ndarray::{AtlasNdError, NDArray};

#[test]
fn array_reductions_match_expected_values() {
    let array = NDArray::from_vec(vec![2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();

    assert_eq!(array.sum(), 21);
    assert_eq!(array.prod(), 720);
    assert_eq!(array.min().unwrap(), 1);
    assert_eq!(array.max().unwrap(), 6);
    assert_eq!(array.mean().unwrap(), 3.5);
}

#[test]
fn scalar_reductions_return_scalar_values() {
    let array = NDArray::from_shape_vec([], vec![9_i32]).unwrap();

    assert_eq!(array.sum(), 9);
    assert_eq!(array.prod(), 9);
    assert_eq!(array.min().unwrap(), 9);
    assert_eq!(array.max().unwrap(), 9);
    assert_eq!(array.mean().unwrap(), 9.0);
}

#[test]
fn view_reductions_support_strided_layouts() {
    let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let view = array.view().transpose();

    assert_eq!(view.sum(), 15);
    assert_eq!(view.min().unwrap(), 0);
    assert_eq!(view.max().unwrap(), 5);
    assert_eq!(view.mean().unwrap(), 2.5);
}

#[test]
fn empty_reductions_return_explicit_errors_when_needed() {
    let array = NDArray::<i32>::new(vec![0, 2], 1);

    assert_eq!(array.sum(), 0);
    assert_eq!(array.prod(), 1);
    assert_eq!(array.min().unwrap_err(), AtlasNdError::EmptyReduction { op: "min" });
    assert_eq!(array.max().unwrap_err(), AtlasNdError::EmptyReduction { op: "max" });
    assert_eq!(array.mean().unwrap_err(), AtlasNdError::EmptyReduction { op: "mean" });
}

#[test]
fn axis_reduction_empty_and_scalar_semantics_are_stable() {
    let empty_axis = NDArray::<i32>::new([0, 3], 1);
    let one_dim = NDArray::from_shape_vec([4], vec![1_i32, 2, 3, 4]).unwrap();
    let zero_lane = NDArray::<i32>::new([2, 0, 3], 1);

    assert_eq!(empty_axis.sum_axis(0).unwrap().data(), &[0, 0, 0]);
    assert_eq!(empty_axis.prod_axis(0).unwrap().data(), &[1, 1, 1]);
    assert_eq!(empty_axis.min_axis(0).unwrap_err(), AtlasNdError::EmptyReduction { op: "min" });
    assert_eq!(empty_axis.max_axis(0).unwrap_err(), AtlasNdError::EmptyReduction { op: "max" });
    assert_eq!(empty_axis.mean_axis(0).unwrap_err(), AtlasNdError::EmptyReduction { op: "mean" });

    assert_eq!(one_dim.sum_axis(-1).unwrap().shape(), &[] as &[usize]);
    assert_eq!(one_dim.prod_axis(-1).unwrap().data(), &[24]);
    assert_eq!(one_dim.min_axis(-1).unwrap().data(), &[1]);
    assert_eq!(one_dim.max_axis(-1).unwrap().data(), &[4]);
    assert_eq!(one_dim.mean_axis(-1).unwrap().data(), &[2.5]);

    assert_eq!(zero_lane.sum_axis(0).unwrap().shape(), &[0, 3]);
    assert!(zero_lane.sum_axis(0).unwrap().data().is_empty());
    assert_eq!(zero_lane.mean_axis(2).unwrap().shape(), &[2, 0]);
    assert!(zero_lane.mean_axis(2).unwrap().data().is_empty());
}

#[test]
fn empty_boundary_slices_preserve_reduction_semantics() {
    let array = NDArray::from_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let view = array.view().slice([2, 3], [0, 0]).unwrap();

    assert_eq!(view.sum(), 0);
    assert_eq!(view.prod(), 1);
    assert_eq!(view.min().unwrap_err(), AtlasNdError::EmptyReduction { op: "min" });
    assert_eq!(view.max().unwrap_err(), AtlasNdError::EmptyReduction { op: "max" });
    assert_eq!(view.mean().unwrap_err(), AtlasNdError::EmptyReduction { op: "mean" });
}

#[test]
fn empty_boundary_slices_preserve_axis_reduction_semantics() {
    let array = NDArray::from_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let view = array.view().slice([2, 3], [0, 0]).unwrap();

    assert_eq!(view.sum_axis(0).unwrap().shape(), &[0]);
    assert!(view.sum_axis(0).unwrap().data().is_empty());
    assert_eq!(view.prod_axis(1).unwrap().shape(), &[0]);
    assert!(view.prod_axis(1).unwrap().data().is_empty());
    assert_eq!(view.min_axis(0).unwrap_err(), AtlasNdError::EmptyReduction { op: "min" });
    assert_eq!(view.max_axis(1).unwrap_err(), AtlasNdError::EmptyReduction { op: "max" });
    assert_eq!(view.mean_axis(0).unwrap_err(), AtlasNdError::EmptyReduction { op: "mean" });
}
