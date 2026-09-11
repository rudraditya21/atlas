use atlas_ndarray::{AtlasNdError, NDArray};

#[test]
fn zip_map_indexed_passes_contiguous_logical_indices() {
    let lhs = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let rhs = NDArray::from_shape_vec([2, 3], vec![10_i32, 11, 12, 13, 14, 15]).unwrap();

    let mapped = lhs
        .zip_map_indexed(&rhs, |index, left, right| {
            (index[0] * 100 + index[1] * 10) as i32 + left + right
        })
        .unwrap();

    assert_eq!(mapped.shape(), &[2, 3]);
    assert_eq!(mapped.data(), &[10, 22, 34, 116, 128, 140]);
    assert!(mapped.is_contiguous());
}

#[test]
fn zip_map_indexed_uses_view_logical_order_and_indices() {
    let lhs = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let rhs = NDArray::from_shape_vec([2, 3], vec![10_i32, 11, 12, 13, 14, 15]).unwrap();

    let mapped = lhs
        .view()
        .transpose()
        .zip_map_indexed(&rhs.view().transpose(), |index, left, right| {
            (index[0] * 10 + index[1]) as i32 + left + right
        })
        .unwrap();

    assert_eq!(mapped.shape(), &[3, 2]);
    assert_eq!(mapped.data(), &[10, 17, 22, 29, 34, 41]);
    assert!(mapped.is_contiguous());
}

#[test]
fn zip_map_indexed_uses_sliced_view_indices() {
    let lhs = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let rhs = NDArray::from_shape_vec([2, 3], vec![10_i32, 11, 12, 13, 14, 15]).unwrap();
    let lhs = lhs.view().slice([0, 1], [2, 2]).unwrap();
    let rhs = rhs.view().slice([0, 1], [2, 2]).unwrap();

    let mapped = lhs
        .zip_map_indexed(&rhs, |index, left, right| {
            (index[0] * 10 + index[1]) as i32 + left * right
        })
        .unwrap();

    assert_eq!(mapped.shape(), &[2, 2]);
    assert_eq!(mapped.data(), &[11, 25, 66, 86]);
    assert!(mapped.is_contiguous());
}

#[test]
fn zip_map_indexed_preserves_scalar_and_empty_shapes() {
    let scalar_lhs = NDArray::from_shape_vec([], vec![2_i32]).unwrap();
    let scalar_rhs = NDArray::from_shape_vec([], vec![3_i32]).unwrap();
    let empty_lhs = NDArray::<i32>::zeros([2, 0, 3]).unwrap();
    let empty_rhs = NDArray::<i32>::zeros([2, 0, 3]).unwrap();

    let mapped_scalar = scalar_lhs
        .zip_map_indexed(&scalar_rhs, |index, left, right| index.len() as i32 + left * right)
        .unwrap();
    let mapped_empty =
        empty_lhs.zip_map_indexed(&empty_rhs, |_, left, right| left + right).unwrap();

    assert_eq!(mapped_scalar.shape(), &[] as &[usize]);
    assert_eq!(mapped_scalar.data(), &[6]);
    assert_eq!(mapped_empty.shape(), &[2, 0, 3]);
    assert!(mapped_empty.data().is_empty());
    assert!(mapped_empty.is_contiguous());
}

#[test]
fn zip_map_indexed_rejects_different_shapes() {
    let lhs = NDArray::from_shape_vec([2, 2], vec![0_i32, 1, 2, 3]).unwrap();
    let rhs = NDArray::from_shape_vec([4], vec![0_i32, 1, 2, 3]).unwrap();

    assert_eq!(
        lhs.zip_map_indexed(&rhs, |_, left, right| left + right).unwrap_err(),
        AtlasNdError::InvalidArgument { op: "zip_map_indexed", reason: "array shapes must match" }
    );
}
