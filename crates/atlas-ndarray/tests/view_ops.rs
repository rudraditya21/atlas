use atlas_ndarray::{AtlasNdError, array::NDArray};

#[test]
fn slicing_and_indexing_preserve_underlying_mapping() {
    let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let slice = array.view().slice([0, 1], vec![2, 2]).unwrap();

    assert_eq!(*slice.get(&[0, 0]).unwrap(), 1);
    assert_eq!(*slice.get(&[1, 1]).unwrap(), 5);
}

#[test]
fn transpose_reorders_metadata_without_copying() {
    let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let transposed = array.view().transpose();

    assert_eq!(transposed.shape(), &[3, 2]);
    assert_eq!(transposed.strides(), &[1, 3]);
    assert_eq!(*transposed.get(&[1, 1]).unwrap(), 4);
}

#[test]
fn transpose_and_reshape_handle_scalar_and_zero_length_views() {
    let scalar = NDArray::new([], 42_i32);
    let zero_length = NDArray::<i32>::zeros([2, 0, 3]);

    let scalar_transposed = scalar.view().transpose();
    let zero_length_reshaped = zero_length.view().reshape([0]).unwrap();

    assert_eq!(scalar_transposed.shape(), &[] as &[usize]);
    assert_eq!(*scalar_transposed.get(&[]).unwrap(), 42);
    assert_eq!(zero_length_reshaped.shape(), &[0]);
    assert!(zero_length_reshaped.is_empty());
}

#[test]
fn reshape_rejects_non_contiguous_views() {
    let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let slice = array.view().slice([0, 1], vec![2, 2]).unwrap();
    let error = slice.reshape(vec![4]).unwrap_err();

    assert_eq!(
        error,
        AtlasNdError::InvalidReshape {
            from: vec![2, 2],
            to: vec![4],
            reason: "only contiguous views can be reshaped",
        }
    );
}

#[test]
fn zero_length_slices_are_allowed_at_axis_boundaries() {
    let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let slice = array.view().slice([2, 3], [0, 0]).unwrap();

    assert_eq!(slice.shape(), &[0, 0]);
    assert!(slice.is_empty());
}
