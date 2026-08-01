use atlas_ndarray::{array::NDArray, AtlasNdError};

#[test]
fn from_vec_returns_shape_mismatch_error() {
    let error = NDArray::from_vec(vec![2, 2], vec![1_i32, 2, 3]).unwrap_err();

    assert_eq!(
        error,
        AtlasNdError::ShapeMismatch {
            expected: 4,
            actual: 3,
        }
    );
}

#[test]
fn get_returns_dimension_mismatch_error() {
    let array = NDArray::new(vec![2, 3], 0_i32);
    let error = array.get(&[0]).unwrap_err();

    assert_eq!(
        error,
        AtlasNdError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );
}

#[test]
fn get_returns_out_of_bounds_error() {
    let array = NDArray::new(vec![2, 3], 0_i32);
    let error = array.get(&[0, 3]).unwrap_err();

    assert_eq!(
        error,
        AtlasNdError::IndexOutOfBounds {
            axis: 1,
            index: 3,
            dim: 3,
        }
    );
}
