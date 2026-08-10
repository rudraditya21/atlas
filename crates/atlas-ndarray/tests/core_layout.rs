use atlas_ndarray::{NDArray, compute_strides};

#[test]
fn compute_strides_matches_row_major_layout() {
    assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
}

#[test]
fn owned_arrays_expose_consistent_layout_metadata() {
    let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();

    assert_eq!(array.shape(), &[2, 3]);
    assert_eq!(array.strides(), &[3, 1]);
    assert_eq!(array.ndim(), 2);
    assert_eq!(array.len(), 6);
    assert!(array.is_contiguous());
}

#[test]
fn indexing_follows_row_major_layout() {
    let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();

    assert_eq!(*array.get(&[0, 2]).unwrap(), 2);
    assert_eq!(*array.get(&[1, 1]).unwrap(), 4);
}

#[test]
fn indexing_supports_negative_indices_consistently() {
    let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();

    assert_eq!(*array.get(&[-1, -1]).unwrap(), 5);
    assert_eq!(*array.get(&[-2, 1]).unwrap(), 1);
}
