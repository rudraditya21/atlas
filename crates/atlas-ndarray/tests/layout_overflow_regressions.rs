use atlas_ndarray::{AtlasNdError, NDArray, checked_compute_strides, checked_element_count};

#[test]
fn checked_shape_products_preserve_zero_extent_and_exact_limit_boundaries() {
    assert_eq!(checked_element_count(&[usize::MAX, 1]).unwrap(), usize::MAX);
    assert_eq!(checked_element_count(&[usize::MAX, 0, usize::MAX]).unwrap(), 0);
    assert_eq!(
        checked_element_count(&[usize::MAX, 2]).unwrap_err(),
        AtlasNdError::ShapeOverflow { op: "element count", shape: vec![usize::MAX, 2] }
    );
}

#[test]
fn checked_strides_handle_zero_extents_and_reject_overflow() {
    assert_eq!(
        checked_compute_strides(&[usize::MAX, 0, usize::MAX]).unwrap(),
        vec![0, usize::MAX, 1]
    );
    assert_eq!(
        checked_compute_strides(&[2, usize::MAX, 2]).unwrap_err(),
        AtlasNdError::ShapeOverflow { op: "stride computation", shape: vec![2, usize::MAX, 2] }
    );
}

#[test]
fn multidimensional_slices_preserve_checked_logical_offsets() {
    let array = NDArray::from_shape_vec([2, 3, 4], (0_i32..24).collect()).unwrap();
    let view = array.view().slice([1, 2, 3], [1, 1, 1]).unwrap();

    assert_eq!(view.offset(), 23);
    assert_eq!(view.item().unwrap(), 23);
}

#[test]
fn empty_slices_accept_boundary_offsets_without_exposing_storage() {
    let array = NDArray::from_shape_vec([2, 3], (0_i32..6).collect()).unwrap();
    let boundary_empty = array.view().slice([2, 3], [0, 0]).unwrap();
    let suffix_empty = array.view().slice([1, 3], [1, 0]).unwrap();

    assert!(boundary_empty.is_empty());
    assert_eq!(boundary_empty.offset(), 0);
    assert!(boundary_empty.is_contiguous());
    assert_eq!(boundary_empty.dense_slice(), Some(&[][..]));
    assert!(suffix_empty.is_empty());
    assert_eq!(suffix_empty.offset(), 3);
    assert_eq!(suffix_empty.dense_slice(), Some(&[][..]));
}
