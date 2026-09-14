use atlas_ndarray::NDArray;

#[test]
fn map_indexed_passes_contiguous_logical_indices() {
    let array = NDArray::from_shape_vec([2, 2], vec![1_i32, 2, 3, 4]).unwrap();

    let mapped = array.map_indexed(|index, value| index[0] as i32 * 10 + index[1] as i32 + value);

    assert_eq!(mapped.shape(), &[2, 2]);
    assert_eq!(mapped.data(), &[1, 3, 13, 15]);
}

#[test]
fn map_indexed_uses_transposed_view_indices() {
    let array = NDArray::from_shape_vec([2, 3], (0_i32..6).collect()).unwrap();

    let mapped = array
        .view()
        .transpose()
        .map_indexed(|index, value| index[0] as i32 * 100 + index[1] as i32 * 10 + value);

    assert_eq!(mapped.shape(), &[3, 2]);
    assert_eq!(mapped.data(), &[0, 13, 101, 114, 202, 215]);
}

#[test]
fn map_indexed_uses_sliced_view_indices() {
    let array = NDArray::from_shape_vec([2, 3], (0_i32..6).collect()).unwrap();
    let view = array.view().slice([0, 1], [2, 2]).unwrap();

    let mapped = view.map_indexed(|index, value| index[0] as i32 * 10 + index[1] as i32 + value);

    assert_eq!(mapped.shape(), &[2, 2]);
    assert_eq!(mapped.data(), &[1, 3, 14, 16]);
}

#[test]
fn map_indexed_preserves_scalar_and_empty_shapes() {
    let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();
    let empty = NDArray::<i32>::zeros([0, 2]).unwrap();
    let mut calls = 0;

    let mapped_scalar = scalar.map_indexed(|index, value| index.len() as i32 + value);
    let mapped_empty = empty.map_indexed(|_, value| {
        calls += 1;
        value
    });

    assert_eq!(mapped_scalar.shape(), &[] as &[usize]);
    assert_eq!(mapped_scalar.data(), &[7]);
    assert_eq!(mapped_empty.shape(), &[0, 2]);
    assert!(mapped_empty.data().is_empty());
    assert_eq!(calls, 0);
}
