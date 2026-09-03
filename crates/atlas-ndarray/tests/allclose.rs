use atlas_ndarray::{AtlasNdError, NDArray, allclose};

#[test]
fn allclose_applies_relative_and_absolute_tolerances() {
    let lhs = NDArray::from_shape_vec([3], vec![1.0_f64, 1000.0, 0.001]).unwrap();
    let rhs = NDArray::from_shape_vec([3], vec![1.000_001_f64, 1000.1, 0.001_01]).unwrap();

    assert!(allclose(&lhs, &rhs, 1.0e-4, 2.0e-5, false).unwrap());
    assert!(!allclose(&lhs, &rhs, 1.0e-8, 0.0, false).unwrap());
}

#[test]
fn allclose_supports_broadcasted_and_strided_view_operands() {
    let lhs_source =
        NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();
    let lhs = lhs_source.view().transpose();
    let rhs = NDArray::from_shape_vec([3, 1], vec![1.0_f64, 2.0, 3.0]).unwrap();

    assert!(allclose(&lhs, &rhs, 0.0, 0.0, false).unwrap());
}

#[test]
fn allclose_handles_nan_and_infinity_explicitly() {
    let lhs = NDArray::from_shape_vec([3], vec![f64::NAN, f64::INFINITY, 1.0]).unwrap();
    let rhs = NDArray::from_shape_vec([3], vec![f64::NAN, f64::INFINITY, 1.0]).unwrap();

    assert!(!allclose(&lhs, &rhs, 0.0, 0.0, false).unwrap());
    assert!(allclose(&lhs, &rhs, 0.0, 0.0, true).unwrap());
}

#[test]
fn allclose_rejects_invalid_tolerances_and_incompatible_shapes() {
    let lhs = NDArray::from_shape_vec([2], vec![1.0_f64, 2.0]).unwrap();
    let rhs = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();

    assert_eq!(
        allclose(&lhs, &rhs, -1.0, 0.0, false).unwrap_err(),
        AtlasNdError::InvalidArgument {
            op: "allclose",
            reason: "tolerances must be finite and non-negative",
        }
    );
    assert!(allclose(&lhs, &rhs, 0.0, 0.0, false).is_err());
}
