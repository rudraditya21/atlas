use atlas_ndarray::{AtlasNdError, NDArray};

#[test]
fn broadcast_traversal_handles_scalar_operands() {
    let values = NDArray::from_shape_vec([2, 2], vec![1_i32, 2, 3, 4]).unwrap();
    let scalar = NDArray::from_shape_vec([], vec![10_i32]).unwrap();

    let result = values.add(&scalar).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.data(), &[11, 12, 13, 14]);
}

#[test]
fn broadcast_traversal_matches_scalar_reference_for_singleton_axes() {
    let lhs_values = [0_i32, 1, 2, 3, 4, 5];
    let rhs_values = [10_i32, 20, 30, 40];
    let lhs = NDArray::from_shape_vec([2, 1, 3], lhs_values.to_vec()).unwrap();
    let rhs = NDArray::from_shape_vec([1, 4, 1], rhs_values.to_vec()).unwrap();
    let expected: Vec<_> = (0..2)
        .flat_map(|batch| {
            (0..4).flat_map(move |column| {
                (0..3).map(move |feature| lhs_values[batch * 3 + feature] + rhs_values[column])
            })
        })
        .collect();

    let result = lhs.add(&rhs).unwrap();

    assert_eq!(result.shape(), &[2, 4, 3]);
    assert_eq!(result.data(), expected);
}

#[test]
fn broadcast_traversal_preserves_zero_length_outputs() {
    let lhs = NDArray::<i32>::zeros([2, 0, 3]).unwrap();
    let rhs = NDArray::<i32>::zeros([1, 0, 1]).unwrap();

    let result = lhs.add(&rhs).unwrap();

    assert_eq!(result.shape(), &[2, 0, 3]);
    assert!(result.is_empty());
}

#[test]
fn broadcast_traversal_preserves_rank_mismatch_errors() {
    let lhs = NDArray::<i32>::zeros([2, 3]).unwrap();
    let rhs = NDArray::<i32>::zeros([2]).unwrap();

    assert_eq!(
        lhs.add(&rhs).unwrap_err(),
        AtlasNdError::InvalidBroadcast {
            lhs: vec![2, 3],
            rhs: vec![2],
            axis: 1,
            lhs_dim: 3,
            rhs_dim: 2,
        }
    );
}
