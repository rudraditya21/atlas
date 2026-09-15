use atlas_ndarray::{AtlasNdError, NDArray, SliceRange};

#[test]
fn map_preserves_contiguous_array_shape_and_values() {
    let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let mapped = array.map(|value| value as f64 + 0.5);

    assert_eq!(mapped.shape(), &[2, 3]);
    assert_eq!(mapped.data(), &[0.5, 1.5, 2.5, 3.5, 4.5, 5.5]);
    assert!(mapped.is_contiguous());
}

#[test]
fn contiguous_map_matches_scalar_logical_traversal() {
    let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let view = array.view().slice([1, 0], [1, 3]).unwrap();

    let expected_array: Vec<_> = array.view().iter().copied().map(|value| value * 3 - 1).collect();
    let expected_view: Vec<_> = view.iter().copied().map(|value| value * 3 - 1).collect();

    assert_eq!(array.map(|value| value * 3 - 1).data(), expected_array.as_slice());
    assert_eq!(view.map(|value| value * 3 - 1).data(), expected_view.as_slice());
}

#[test]
fn map_materializes_transposed_views_in_logical_order() {
    let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let mapped = array.view().transpose().map(|value| value * 2);

    assert_eq!(mapped.shape(), &[3, 2]);
    assert_eq!(mapped.data(), &[0, 6, 2, 8, 4, 10]);
    assert!(mapped.is_contiguous());
}

#[test]
fn map_materializes_sliced_views_in_logical_order() {
    let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let mapped = array.view().slice([0, 1], [2, 2]).unwrap().map(|value| value > 2);

    assert_eq!(mapped.shape(), &[2, 2]);
    assert_eq!(mapped.data(), &[false, false, true, true]);
    assert!(mapped.is_contiguous());
}

#[test]
fn strided_map_matches_scalar_logical_traversal() {
    let source = NDArray::from_shape_vec([2, 4], (0_i32..8).collect()).unwrap();
    let row_strided = source.view().slice([0, 0], [2, 3]).unwrap();
    let transposed = source.view().transpose();
    let stepped = source
        .view()
        .slice_ranges([SliceRange::full(), SliceRange::new(Some(0), Some(4), 2)])
        .unwrap();

    for view in [&row_strided, &transposed, &stepped] {
        let expected: Vec<_> = view.iter().copied().map(|value| value * 2 + 1).collect();
        let mapped = view.map(|value| value * 2 + 1);

        assert_eq!(mapped.data(), expected.as_slice());
    }
}

#[test]
fn reverse_slices_remain_rejected_until_negative_steps_are_supported() {
    let array = NDArray::from_shape_vec([4], vec![0_i32, 1, 2, 3]).unwrap();

    assert_eq!(
        array.view().slice_ranges([SliceRange::new(None, None, -1)]).unwrap_err(),
        AtlasNdError::InvalidArgument { op: "slice", reason: "negative step is not supported" }
    );
}

#[test]
fn map_preserves_scalar_and_empty_shapes() {
    let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();
    let empty = NDArray::<i32>::zeros([2, 0, 3]).unwrap();

    let mapped_scalar = scalar.map(|value| value == 7);
    let mapped_empty = empty.view().map(|value| value + 1);

    assert_eq!(mapped_scalar.shape(), &[] as &[usize]);
    assert_eq!(mapped_scalar.data(), &[true]);
    assert_eq!(mapped_empty.shape(), &[2, 0, 3]);
    assert!(mapped_empty.data().is_empty());
    assert!(mapped_empty.is_contiguous());
}
