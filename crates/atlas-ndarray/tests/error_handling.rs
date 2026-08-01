use atlas_ndarray::{AtlasNdError, array::NDArray};

#[test]
fn from_vec_returns_shape_mismatch_error() {
    let error = NDArray::from_vec(vec![2, 2], vec![1_i32, 2, 3]).unwrap_err();

    assert_eq!(error, AtlasNdError::ShapeMismatch { expected: 4, actual: 3 });
}

#[test]
fn get_returns_dimension_mismatch_error() {
    let array = NDArray::new(vec![2, 3], 0_i32);
    let error = array.get(&[0]).unwrap_err();

    assert_eq!(error, AtlasNdError::DimensionMismatch { expected: 2, actual: 1 });
}

#[test]
fn get_returns_out_of_bounds_error() {
    let array = NDArray::new(vec![2, 3], 0_i32);
    let error = array.get(&[0, 3]).unwrap_err();

    assert_eq!(error, AtlasNdError::IndexOutOfBounds { axis: 1, index: 3, dim: 3 });
}

#[test]
fn get_mut_returns_consistent_indexing_errors() {
    let mut array = NDArray::new([2, 2], 0_i32);

    assert_eq!(
        array.get_mut(&[0]).unwrap_err(),
        AtlasNdError::DimensionMismatch { expected: 2, actual: 1 }
    );
    assert_eq!(
        array.get_mut(&[0, 2]).unwrap_err(),
        AtlasNdError::IndexOutOfBounds { axis: 1, index: 2, dim: 2 }
    );
}

#[test]
fn slice_and_reshape_return_stable_view_errors() {
    let array = NDArray::new([2, 3], 0_i32);
    let view = array.view();

    assert_eq!(
        view.slice([0, 0], [1]).unwrap_err(),
        AtlasNdError::DimensionMismatch { expected: 2, actual: 1 }
    );
    assert_eq!(
        array.view().reshape([5]).unwrap_err(),
        AtlasNdError::InvalidReshape {
            from: vec![2, 3],
            to: vec![5],
            reason: "element count must remain unchanged",
        }
    );
}
