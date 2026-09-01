use atlas_ndarray::{AtlasNdError, NDArray};

#[test]
fn split_returns_equal_sized_views_with_expected_boundaries() {
    let array = NDArray::from_shape_vec([2, 6], (0_i32..12).collect()).unwrap();
    let parts = array.split(3, 1).unwrap();

    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].shape(), &[2, 2]);
    assert_eq!(parts[0].strides(), &[6, 1]);
    assert_eq!(parts[0].offset(), 0);
    assert_eq!(parts[1].offset(), 2);
    assert_eq!(parts[2].offset(), 4);
    assert_eq!(parts[0].to_owned().data(), &[0, 1, 6, 7]);
    assert_eq!(parts[2].to_owned().data(), &[4, 5, 10, 11]);
    assert!(parts.iter().all(|part| !part.is_owned()));
}

#[test]
fn array_split_distributes_remainder_to_earlier_views() {
    let array = NDArray::from_shape_vec([7], (0_i32..7).collect()).unwrap();
    let parts = array.array_split(3, 0).unwrap();

    assert_eq!(parts.iter().map(|part| part.shape()[0]).collect::<Vec<_>>(), vec![3, 2, 2]);
    assert_eq!(parts[0].to_owned().data(), &[0, 1, 2]);
    assert_eq!(parts[1].to_owned().data(), &[3, 4]);
    assert_eq!(parts[2].to_owned().data(), &[5, 6]);
}

#[test]
fn array_split_preserves_empty_outputs_and_supports_negative_axes() {
    let vector = NDArray::from_shape_vec([2], vec![4_i32, 5]).unwrap();
    let empty_parts = vector.array_split(4, -1).unwrap();
    let matrix = NDArray::from_shape_vec([2, 4], (0_i32..8).collect()).unwrap();
    let negative_axis_parts = matrix.split(2, -1).unwrap();

    assert_eq!(
        empty_parts.iter().map(|part| part.shape()[0]).collect::<Vec<_>>(),
        vec![1, 1, 0, 0]
    );
    assert_eq!(empty_parts[2].to_owned().data(), &[] as &[i32]);
    assert_eq!(negative_axis_parts[0].shape(), &[2, 2]);
    assert_eq!(negative_axis_parts[1].to_owned().data(), &[2, 3, 6, 7]);
}

#[test]
fn split_validates_sections_divisibility_and_axes() {
    let array = NDArray::from_shape_vec([5], (0_i32..5).collect()).unwrap();

    assert_eq!(
        array.split(0, 0).unwrap_err(),
        AtlasNdError::InvalidArgument { op: "split", reason: "sections must be greater than zero" }
    );
    assert_eq!(
        array.split(2, 0).unwrap_err(),
        AtlasNdError::InvalidArgument {
            op: "split",
            reason: "axis length must be divisible by sections",
        }
    );
    assert_eq!(
        array.array_split(2, 1).unwrap_err(),
        AtlasNdError::InvalidAxis { axis: 1, ndim: 1 }
    );
}

#[test]
fn concatenating_split_views_reconstructs_logical_order() {
    let array = NDArray::from_shape_vec([2, 7], (0_i32..14).collect()).unwrap();
    let parts = array.array_split(3, 1).unwrap();
    let reconstructed = NDArray::concatenate(&parts, -1).unwrap();

    assert_eq!(reconstructed.shape(), array.shape());
    assert_eq!(reconstructed.data(), array.data());
}
