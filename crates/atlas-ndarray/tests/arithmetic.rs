use atlas_ndarray::{AtlasNdError, NDArray, Numeric};

fn assert_array_eq<T>(lhs: &NDArray<T>, rhs: &NDArray<T>)
where
    T: Numeric + PartialEq,
{
    assert_eq!(lhs.shape(), rhs.shape());
    assert_eq!(lhs.strides(), rhs.strides());
    assert_eq!(lhs.data(), rhs.data());
    assert_eq!(lhs.is_contiguous(), rhs.is_contiguous());
}

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
    let lhs = NDArray::new(vec![2, 3], 1_i32).unwrap();
    let rhs = NDArray::new(vec![4, 3], 1_i32).unwrap();
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

#[test]
fn scalar_values_match_scalar_shaped_array_results() {
    let array = NDArray::from_vec([2, 2], vec![2_i32, 4, 6, 8]).unwrap();
    let scalar = NDArray::from_shape_vec([], vec![2_i32]).unwrap();

    assert_array_eq(&array.add(2), &array.add(&scalar).unwrap());
    assert_array_eq(&array.sub(2), &array.sub(&scalar).unwrap());
    assert_array_eq(&array.mul(2), &array.mul(&scalar).unwrap());
    assert_array_eq(&array.div(2), &array.div(&scalar).unwrap());
}

#[test]
fn scalar_and_scalar_shaped_array_paths_match_for_empty_outputs() {
    let empty = NDArray::<i32>::zeros([0, 3]).unwrap();
    let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();

    assert_array_eq(&empty.add(7), &empty.add(&scalar).unwrap());
    assert_array_eq(&empty.sub(7), &empty.sub(&scalar).unwrap());
    assert_array_eq(&empty.mul(7), &empty.mul(&scalar).unwrap());
    assert_array_eq(&empty.div(7), &empty.div(&scalar).unwrap());
}
