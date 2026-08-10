use atlas_ndarray::{AsArray, AtlasNdError, NDArray};

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
    let scalar = NDArray::new([], 42_i32).unwrap();
    let zero_length = NDArray::<i32>::zeros([2, 0, 3]).unwrap();

    let scalar_transposed = scalar.view().transpose();
    let zero_length_reshaped = zero_length.view().reshape([0]).unwrap();

    assert_eq!(scalar_transposed.shape(), &[] as &[usize]);
    assert_eq!(*scalar_transposed.get(&[] as &[i64]).unwrap(), 42);
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
            reason: "only contiguous or empty views can be reshaped",
        }
    );
}

#[test]
fn reshape_allows_zero_length_non_contiguous_views() {
    let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let empty = array.view().slice([2, 3], [0, 0]).unwrap();
    let reshaped = empty.reshape([0]).unwrap();

    assert_eq!(reshaped.shape(), &[0]);
    assert_eq!(reshaped.strides(), &[1]);
    assert_eq!(reshaped.offset(), 0);
    assert!(reshaped.is_empty());
}

#[test]
fn chained_transpose_slice_and_empty_reshape_preserve_offset_and_mapping() {
    let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let chained = array.view().transpose().slice([1, 2], [2, 0]).unwrap();
    let reshaped = chained.clone().reshape([0]).unwrap();

    assert_eq!(chained.shape(), &[2, 0]);
    assert_eq!(chained.strides(), &[1, 3]);
    assert_eq!(chained.offset(), 1);
    assert!(chained.is_empty());

    assert_eq!(reshaped.shape(), &[0]);
    assert_eq!(reshaped.strides(), &[1]);
    assert_eq!(reshaped.offset(), 1);
    assert!(reshaped.is_empty());
}

#[test]
fn zero_length_slices_are_allowed_at_axis_boundaries() {
    let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let slice = array.view().slice([2, 3], [0, 0]).unwrap();

    assert_eq!(slice.shape(), &[0, 0]);
    assert!(slice.is_empty());
}

#[test]
fn empty_boundary_slices_expose_stable_public_view_metadata() {
    let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let slice = array.view().slice([2, 3], [0, 0]).unwrap();

    assert_eq!(slice.shape(), &[0, 0]);
    assert_eq!(slice.strides(), &[3, 1]);
    assert_eq!(slice.len(), 0);
    assert!(slice.is_empty());
    assert_eq!(slice.ndim(), 2);
    assert!(!slice.is_contiguous());
}

#[test]
fn ravel_preserves_views_when_possible_and_materializes_when_needed() {
    let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let contiguous = array.ravel();
    let transposed = array.view().transpose().ravel();

    assert!(matches!(&contiguous, AsArray::Borrowed(_)));
    assert_eq!(contiguous.view().shape(), &[6]);
    assert_eq!(contiguous.view().strides(), &[1]);
    assert_eq!(contiguous.view().data(), array.data());

    assert!(matches!(&transposed, AsArray::Owned(_)));
    assert_eq!(transposed.view().shape(), &[6]);
    assert_eq!(transposed.view().strides(), &[1]);
    assert_eq!(transposed.into_owned().data(), &[0, 3, 1, 4, 2, 5]);
}

#[test]
fn flatten_always_returns_owned_flattened_copies() {
    let mut array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let contiguous = array.flatten();
    let transposed = array.view().transpose().flatten();

    *array.get_mut(&[0, 0]).unwrap() = 99;

    assert_eq!(contiguous.shape(), &[6]);
    assert_eq!(contiguous.strides(), &[1]);
    assert_eq!(contiguous.data(), &[0, 1, 2, 3, 4, 5]);

    assert_eq!(transposed.shape(), &[6]);
    assert_eq!(transposed.strides(), &[1]);
    assert_eq!(transposed.data(), &[0, 3, 1, 4, 2, 5]);
}

#[test]
fn squeeze_and_expand_dims_preserve_shape_mapping_through_singleton_axes() {
    let array = NDArray::from_vec(vec![2, 1, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let squeezed = array.squeeze();
    let expanded = squeezed.clone().expand_dims(1).unwrap();

    assert_eq!(squeezed.shape(), &[2, 3]);
    assert_eq!(squeezed.strides(), &[3, 1]);
    assert_eq!(*squeezed.get(&[1, 2]).unwrap(), 5);

    assert_eq!(expanded.shape(), &[2, 1, 3]);
    assert_eq!(expanded.strides(), &[3, 3, 1]);
    assert_eq!(*expanded.get(&[1, 0, 2]).unwrap(), 5);
}

#[test]
fn contiguous_slice_then_reshape_preserves_offset_and_logical_order() {
    let array = NDArray::from_vec(vec![2, 3, 4], (0_i32..24).collect()).unwrap();
    let sliced = array.view().slice([1, 0, 0], [1, 3, 4]).unwrap();
    let reshaped = sliced.clone().reshape([12]).unwrap();

    assert_eq!(sliced.shape(), &[1, 3, 4]);
    assert_eq!(sliced.strides(), &[12, 4, 1]);
    assert_eq!(sliced.offset(), 12);
    assert_eq!(*sliced.get(&[0, 2, 3]).unwrap(), 23);

    assert_eq!(reshaped.shape(), &[12]);
    assert_eq!(reshaped.strides(), &[1]);
    assert_eq!(reshaped.offset(), 12);
    assert_eq!(*reshaped.get(&[11]).unwrap(), 23);
}
