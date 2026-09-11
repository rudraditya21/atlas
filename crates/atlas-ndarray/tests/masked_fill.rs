use atlas_ndarray::{AtlasNdError, NDArray};

#[test]
fn masked_fill_replaces_selected_values() {
    let mut values = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let mask =
        NDArray::from_shape_vec([2, 3], vec![true, false, true, false, true, false]).unwrap();

    values.masked_fill(&mask, 9).unwrap();

    assert_eq!(values.data(), &[9, 1, 9, 3, 9, 5]);
}

#[test]
fn masked_fill_supports_strided_mask_views() {
    let mut values = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let mask =
        NDArray::from_shape_vec([3, 2], vec![true, false, false, true, true, false]).unwrap();

    values.masked_fill(&mask.view().transpose(), -1).unwrap();

    assert_eq!(values.data(), &[-1, 1, -1, 3, -1, 5]);
}

#[test]
fn masked_fill_handles_scalar_and_empty_arrays() {
    let mut scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();
    let scalar_mask = NDArray::from_shape_vec([], vec![true]).unwrap();
    let mut empty = NDArray::<i32>::zeros([2, 0, 3]).unwrap();
    let empty_mask = NDArray::from_shape_vec([2, 0, 3], Vec::<bool>::new()).unwrap();

    scalar.masked_fill(&scalar_mask, 9).unwrap();
    empty.masked_fill(&empty_mask, 9).unwrap();

    assert_eq!(scalar.data(), &[9]);
    assert!(empty.data().is_empty());
}

#[test]
fn masked_fill_rejects_masks_with_different_shapes() {
    let mut values = NDArray::from_shape_vec([2, 2], vec![0_i32, 1, 2, 3]).unwrap();
    let mask = NDArray::from_shape_vec([4], vec![true, false, true, false]).unwrap();

    assert_eq!(
        values.masked_fill(&mask, 9).unwrap_err(),
        AtlasNdError::InvalidArgument {
            op: "masked_fill",
            reason: "mask shape must match array shape",
        }
    );
    assert_eq!(values.data(), &[0, 1, 2, 3]);
}
