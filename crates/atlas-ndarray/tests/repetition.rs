use atlas_ndarray::{AtlasNdError, NDArray};

#[test]
fn repeat_preserves_reference_order_along_each_axis() {
    let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let repeat_columns = array.repeat(2, 1).unwrap();
    let repeat_rows = array.repeat(2, 0).unwrap();

    assert_eq!(repeat_columns.shape(), &[2, 6]);
    assert_eq!(repeat_columns.data(), &[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
    assert_eq!(repeat_rows.shape(), &[4, 3]);
    assert_eq!(repeat_rows.data(), &[0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5]);
    assert!(repeat_columns.is_owned());
    assert!(repeat_rows.is_contiguous());
}

#[test]
fn repeat_handles_zero_repeats_and_zero_sized_inputs() {
    let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let zero_repeated = array.repeat(0, -1).unwrap();
    let zero_sized = NDArray::<i32>::from_shape_vec([2, 0, 3], Vec::new()).unwrap();
    let repeated_zero_sized = zero_sized.repeat(2, 1).unwrap();

    assert_eq!(zero_repeated.shape(), &[2, 0]);
    assert!(zero_repeated.data().is_empty());
    assert_eq!(repeated_zero_sized.shape(), &[2, 0, 3]);
    assert!(repeated_zero_sized.data().is_empty());
}

#[test]
fn tile_supports_scalar_and_multi_axis_reference_layouts() {
    let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();
    let matrix = NDArray::from_shape_vec([2, 2], vec![1_i32, 2, 3, 4]).unwrap();
    let scalar_tiled = scalar.tile([3]).unwrap();
    let tiled = matrix.tile([2, 3]).unwrap();

    assert_eq!(scalar_tiled.shape(), &[3]);
    assert_eq!(scalar_tiled.data(), &[7, 7, 7]);
    assert_eq!(tiled.shape(), &[4, 6]);
    assert_eq!(
        tiled.data(),
        &[1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4]
    );
    assert!(tiled.is_owned());
    assert!(tiled.is_contiguous());
}

#[test]
fn tile_handles_zero_sized_inputs_and_validates_empty_repetitions() {
    let zero_sized = NDArray::<i32>::from_shape_vec([2, 0, 3], Vec::new()).unwrap();
    let tiled = zero_sized.tile([2, 3, 4]).unwrap();
    let array = NDArray::from_shape_vec([2], vec![1_i32, 2]).unwrap();

    assert_eq!(tiled.shape(), &[4, 0, 12]);
    assert!(tiled.data().is_empty());
    assert_eq!(
        array.tile([] as [usize; 0]).unwrap_err(),
        AtlasNdError::InvalidArgument { op: "tile", reason: "repetitions must not be empty" }
    );
}

#[test]
fn repeat_rejects_axes_for_scalar_inputs() {
    let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();

    assert_eq!(scalar.repeat(2, 0).unwrap_err(), AtlasNdError::InvalidAxis { axis: 0, ndim: 0 });
}
