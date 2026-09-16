use atlas_ndarray::{AsArray, AtlasNdError, NDArray, SliceRange};

#[test]
fn contiguous_arrays_reshape_and_ravel_without_materializing() {
    let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let reshaped = array.reshape([3, 2]).unwrap();
    let raveled = array.ravel();
    let flattened = array.flatten();

    assert!(!reshaped.is_owned());
    assert_eq!(reshaped.data().as_ptr(), array.data().as_ptr());
    assert_eq!(reshaped.shape(), &[3, 2]);
    assert_eq!(reshaped.strides(), &[2, 1]);
    assert!(matches!(&raveled, AsArray::Borrowed(_)));
    assert_eq!(raveled.view().data().as_ptr(), array.data().as_ptr());
    assert!(flattened.is_owned());
    assert_eq!(flattened.data(), array.data());
    assert_ne!(flattened.data().as_ptr(), array.data().as_ptr());
}

#[test]
fn transposed_and_sliced_views_materialize_logical_values() {
    let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let transposed = array.view().transpose();
    let sliced = array.view().slice([0, 1], [2, 2]).unwrap();

    assert_eq!(
        transposed.clone().reshape([6]).unwrap_err(),
        AtlasNdError::InvalidReshape {
            from: vec![3, 2],
            to: vec![6],
            reason: "only contiguous or empty views can be reshaped",
        }
    );
    assert_eq!(
        sliced.clone().reshape([4]).unwrap_err(),
        AtlasNdError::InvalidReshape {
            from: vec![2, 2],
            to: vec![4],
            reason: "only contiguous or empty views can be reshaped",
        }
    );
    assert!(matches!(transposed.clone().ravel(), AsArray::Owned(_)));
    assert_eq!(sliced.to_owned().data(), &[1, 2, 4, 5]);
    assert_eq!(transposed.flatten().data(), &[0, 3, 1, 4, 2, 5]);
}

#[test]
fn copy_and_to_owned_always_return_independent_owned_arrays() {
    let array = NDArray::from_shape_vec([2, 2], vec![1_i32, 2, 3, 4]).unwrap();
    let view = array.view().transpose();
    let copy = array.copy();
    let owned = view.to_owned();
    let view_copy = view.copy();

    assert!(copy.is_owned());
    assert_eq!(copy.data(), array.data());
    assert_ne!(copy.data().as_ptr(), array.data().as_ptr());
    assert!(owned.is_owned());
    assert_eq!(owned.data(), &[1, 3, 2, 4]);
    assert_eq!(view_copy.data(), owned.data());
}

#[test]
fn scalar_and_zero_sized_inputs_preserve_materialization_contracts() {
    let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();
    let zero_sized = NDArray::<i32>::from_shape_vec([2, 0, 3], Vec::new()).unwrap();

    let scalar_reshaped = scalar.reshape([1]).unwrap();
    let scalar_flattened = scalar.flatten();
    let empty_reshaped = zero_sized.reshape([0]).unwrap();
    let empty_raveled = zero_sized.ravel();
    let empty_flattened = zero_sized.flatten();

    assert!(!scalar_reshaped.is_owned());
    assert_eq!(scalar_reshaped.data(), &[7]);
    assert!(scalar_flattened.is_owned());
    assert_eq!(scalar_flattened.shape(), &[1]);
    assert_eq!(scalar_flattened.data(), &[7]);
    assert!(!empty_reshaped.is_owned());
    assert_eq!(empty_reshaped.shape(), &[0]);
    assert!(matches!(&empty_raveled, AsArray::Borrowed(_)));
    assert_eq!(empty_flattened.shape(), &[0]);
    assert!(empty_flattened.is_owned());
}

#[test]
fn materialization_preserves_offsets_strides_transposes_and_empty_views() {
    let values = NDArray::from_shape_vec([3, 4], (0_i32..12).collect()).unwrap();
    let offset = values.view().slice([1, 1], [2, 3]).unwrap();
    let stepped = values
        .view()
        .slice_ranges([SliceRange::full(), SliceRange::new(Some(0), Some(4), 2)])
        .unwrap();
    let empty = values.view().slice([3, 0], [0, 4]).unwrap();

    assert_eq!(offset.to_owned().data(), &[5, 6, 7, 9, 10, 11]);
    assert_eq!(stepped.to_owned().data(), &[0, 2, 4, 6, 8, 10]);
    assert_eq!(
        values.view().transpose().to_owned().data(),
        &[0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    );
    assert!(empty.to_owned().is_empty());
}

#[test]
fn materialization_preserves_reverse_slice_rejection_until_reverse_views_exist() {
    let values = NDArray::from_shape_vec([3], vec![0_i32, 1, 2]).unwrap();

    assert_eq!(
        values.view().slice_ranges([SliceRange::new(None, None, -1)]).unwrap_err(),
        AtlasNdError::InvalidArgument { op: "slice", reason: "negative step is not supported" }
    );
}
