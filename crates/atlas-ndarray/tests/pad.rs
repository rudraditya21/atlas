use atlas_ndarray::{AtlasNdError, NDArray};

#[test]
fn pad_applies_constant_values_to_one_dimensional_arrays() {
    let array = NDArray::from_shape_vec([2], vec![1_i32, 2]).unwrap();
    let padded = array.pad([(2, 1)], 0).unwrap();

    assert_eq!(padded.shape(), &[5]);
    assert_eq!(padded.strides(), &[1]);
    assert_eq!(padded.data(), &[0, 0, 1, 2, 0]);
    assert!(padded.is_owned());
    assert!(padded.is_contiguous());
}

#[test]
fn pad_supports_asymmetric_two_dimensional_widths() {
    let array = NDArray::from_shape_vec([2, 2], vec![1_i32, 2, 3, 4]).unwrap();
    let padded = array.pad([(1, 0), (2, 1)], 9).unwrap();

    assert_eq!(padded.shape(), &[3, 5]);
    assert_eq!(padded.strides(), &[5, 1]);
    assert_eq!(padded.data(), &[9, 9, 9, 9, 9, 9, 9, 1, 2, 9, 9, 9, 3, 4, 9]);
}

#[test]
fn pad_preserves_nd_coordinate_mapping() {
    let array = NDArray::from_shape_vec([1, 1, 2], vec![5_i32, 6]).unwrap();
    let padded = array.pad([(0, 1), (1, 0), (1, 1)], -1).unwrap();

    assert_eq!(padded.shape(), &[2, 2, 4]);
    assert_eq!(*padded.get(&[0, 1, 1]).unwrap(), 5);
    assert_eq!(*padded.get(&[0, 1, 2]).unwrap(), 6);
    assert_eq!(*padded.get(&[1, 1, 2]).unwrap(), -1);
    assert_eq!(*padded.get(&[0, 0, 0]).unwrap(), -1);
}

#[test]
fn pad_handles_scalar_and_empty_dimensions() {
    let scalar = NDArray::from_shape_vec([], vec![4_i32]).unwrap();
    let empty = NDArray::<i32>::from_shape_vec([0, 2], Vec::new()).unwrap();
    let padded_scalar = scalar.pad([] as [(usize, usize); 0], 9).unwrap();
    let padded_empty = empty.pad([(1, 1), (1, 0)], 7).unwrap();

    assert_eq!(padded_scalar.shape(), &[] as &[usize]);
    assert_eq!(padded_scalar.data(), &[4]);
    assert_eq!(padded_empty.shape(), &[2, 3]);
    assert_eq!(padded_empty.data(), &[7, 7, 7, 7, 7, 7]);
}

#[test]
fn pad_validates_width_rank() {
    let matrix = NDArray::from_shape_vec([2, 2], vec![1_i32, 2, 3, 4]).unwrap();

    assert_eq!(
        matrix.pad([(1, 1)], 0).unwrap_err(),
        AtlasNdError::DimensionMismatch { expected: 2, actual: 1 }
    );
    assert_eq!(
        matrix.pad([(1, 1), (1, 1), (1, 1)], 0).unwrap_err(),
        AtlasNdError::DimensionMismatch { expected: 2, actual: 3 }
    );
}
