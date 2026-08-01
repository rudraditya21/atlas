use atlas_ndarray::{array::NDArray, AtlasNdError};

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
    assert_eq!(
        array.mean().unwrap_err(),
        AtlasNdError::EmptyReduction { op: "mean" }
    );
}
