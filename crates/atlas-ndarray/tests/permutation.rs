use atlas_ndarray::{AtlasNdError, NDArray};

#[test]
fn permute_axes_reorders_metadata_and_logical_values_without_materializing() {
    let array = NDArray::from_shape_vec([2, 3, 4], (0_i32..24).collect()).unwrap();
    let permuted = array.permute_axes([2, 0, 1]).unwrap();

    assert_eq!(permuted.shape(), &[4, 2, 3]);
    assert_eq!(permuted.strides(), &[1, 12, 4]);
    assert_eq!(*permuted.get(&[3, 1, 2]).unwrap(), 23);
    assert_eq!(permuted.data().as_ptr(), array.data().as_ptr());
    assert!(!permuted.is_owned());
}

#[test]
fn swap_axes_supports_negative_axes_without_materializing() {
    let array = NDArray::from_shape_vec([2, 3, 4], (0_i32..24).collect()).unwrap();
    let swapped = array.swap_axes(-1, 0).unwrap();

    assert_eq!(swapped.shape(), &[4, 3, 2]);
    assert_eq!(swapped.strides(), &[1, 4, 12]);
    assert_eq!(*swapped.get(&[3, 2, 1]).unwrap(), 23);
    assert_eq!(swapped.data().as_ptr(), array.data().as_ptr());
    assert!(!swapped.is_owned());
}

#[test]
fn permutation_preserves_scalar_and_vector_metadata() {
    let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();
    let vector = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();

    let scalar_permuted = scalar.permute_axes([] as [i32; 0]).unwrap();
    let vector_permuted = vector.permute_axes([-1]).unwrap();
    let vector_swapped = vector.swap_axes(0, 0).unwrap();

    assert_eq!(scalar_permuted.shape(), &[] as &[usize]);
    assert_eq!(scalar_permuted.strides(), &[] as &[usize]);
    assert_eq!(*scalar_permuted.get(&[] as &[i32]).unwrap(), 7);
    assert_eq!(vector_permuted.shape(), &[3]);
    assert_eq!(vector_permuted.strides(), &[1]);
    assert_eq!(vector_permuted.data(), vector.data());
    assert_eq!(vector_swapped.shape(), &[3]);
    assert_eq!(vector_swapped.strides(), &[1]);
}

#[test]
fn permutation_validates_rank_duplicate_and_out_of_range_axes() {
    let array = NDArray::from_shape_vec([2, 3, 4], (0_i32..24).collect()).unwrap();

    assert_eq!(
        array.permute_axes([0, 1]).unwrap_err(),
        AtlasNdError::DimensionMismatch { expected: 3, actual: 2 }
    );
    assert_eq!(
        array.permute_axes([0, 0, 1]).unwrap_err(),
        AtlasNdError::InvalidArgument { op: "permute_axes", reason: "axes must be a permutation" }
    );
    assert_eq!(
        array.permute_axes([0, 1, -4]).unwrap_err(),
        AtlasNdError::InvalidAxis { axis: -4, ndim: 3 }
    );
}
