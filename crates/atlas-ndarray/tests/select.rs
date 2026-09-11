use atlas_ndarray::{AtlasNdError, NDArray};

#[test]
fn select_returns_masked_contiguous_values() {
    let values = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let mask =
        NDArray::from_shape_vec([2, 3], vec![true, false, true, false, true, false]).unwrap();

    let selected = values.select(&mask).unwrap();

    assert_eq!(selected.shape(), &[3]);
    assert_eq!(selected.data(), &[0, 2, 4]);
    assert!(selected.is_contiguous());
}

#[test]
fn select_uses_transposed_view_logical_order() {
    let values = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let mask =
        NDArray::from_shape_vec([2, 3], vec![true, false, false, true, true, false]).unwrap();

    let selected = values.view().transpose().select(&mask.view().transpose()).unwrap();

    assert_eq!(selected.shape(), &[3]);
    assert_eq!(selected.data(), &[0, 3, 4]);
    assert!(selected.is_contiguous());
}

#[test]
fn select_uses_sliced_view_logical_order() {
    let values = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let mask =
        NDArray::from_shape_vec([2, 3], vec![false, true, false, false, true, false]).unwrap();
    let values = values.view().slice([0, 1], [2, 2]).unwrap();
    let mask = mask.view().slice([0, 1], [2, 2]).unwrap();

    let selected = values.select(&mask).unwrap();

    assert_eq!(selected.shape(), &[2]);
    assert_eq!(selected.data(), &[1, 4]);
    assert!(selected.is_contiguous());
}

#[test]
fn select_handles_scalar_and_empty_inputs() {
    let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();
    let scalar_mask = NDArray::from_shape_vec([], vec![true]).unwrap();
    let empty = NDArray::<i32>::zeros([2, 0, 3]).unwrap();
    let empty_mask = NDArray::from_shape_vec([2, 0, 3], Vec::<bool>::new()).unwrap();

    let selected_scalar = scalar.select(&scalar_mask).unwrap();
    let selected_empty = empty.view().select(&empty_mask).unwrap();

    assert_eq!(selected_scalar.shape(), &[1]);
    assert_eq!(selected_scalar.data(), &[7]);
    assert_eq!(selected_empty.shape(), &[0]);
    assert!(selected_empty.data().is_empty());
    assert!(selected_empty.is_contiguous());
}

#[test]
fn select_rejects_masks_with_different_shapes() {
    let values = NDArray::from_shape_vec([2, 2], vec![0_i32, 1, 2, 3]).unwrap();
    let mask = NDArray::from_shape_vec([4], vec![true, false, true, false]).unwrap();

    assert_eq!(
        values.select(&mask).unwrap_err(),
        AtlasNdError::InvalidArgument { op: "select", reason: "mask shape must match array shape" }
    );
}
