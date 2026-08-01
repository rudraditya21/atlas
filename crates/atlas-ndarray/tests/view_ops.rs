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
