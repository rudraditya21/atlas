use atlas_ndarray::{AtlasNdError, array::NDArray};

#[test]
fn elementwise_add_broadcasts_singleton_dimensions() {
    let lhs = NDArray::from_vec(vec![2, 1], vec![1_i32, 2]).unwrap();
    let rhs = NDArray::from_vec(vec![1, 3], vec![10_i32, 20, 30]).unwrap();

    let result = lhs.add(&rhs).unwrap();

    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(result.data(), &[11, 21, 31, 12, 22, 32]);
}

#[test]
fn elementwise_mul_supports_scalar_like_inputs() {
    let lhs = NDArray::from_vec(vec![2, 2], vec![1_i32, 2, 3, 4]).unwrap();
    let rhs = NDArray::from_vec(vec![], vec![10_i32]).unwrap();

    let result = lhs.mul(&rhs).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result.data(), &[10, 20, 30, 40]);
}

#[test]
fn elementwise_ops_return_broadcast_errors_for_incompatible_shapes() {
    let lhs = NDArray::new(vec![2, 3], 1_i32);
    let rhs = NDArray::new(vec![4, 3], 1_i32);
    let error = lhs.sub(&rhs).unwrap_err();

    assert_eq!(
        error,
        AtlasNdError::InvalidBroadcast {
            lhs: vec![2, 3],
            rhs: vec![4, 3],
            axis: 0,
            lhs_dim: 2,
            rhs_dim: 4,
        }
    );
}

#[test]
fn elementwise_methods_dispatch_to_scalar_paths() {
    let array = NDArray::from_vec(vec![3], vec![1_i32, 2, 3]).unwrap();

    assert_eq!(array.add(1).data(), &[2, 3, 4]);
    assert_eq!(array.sub(1).data(), &[0, 1, 2]);
    assert_eq!(array.mul(2).data(), &[2, 4, 6]);
    assert_eq!(array.div(2).data(), &[0, 1, 1]);
}
