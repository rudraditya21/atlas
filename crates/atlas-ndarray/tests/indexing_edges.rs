use atlas_ndarray::{AtlasNdError, NDArray};

#[test]
fn nonzero_and_argwhere_define_scalar_empty_singleton_and_zero_length_behavior() {
    let scalar = NDArray::from_shape_vec([], vec![true]).unwrap();
    let empty = NDArray::<i32>::from_shape_vec([0], Vec::new()).unwrap();
    let singleton = NDArray::from_shape_vec([1], vec![7_i32]).unwrap();
    let zero_length_axis = NDArray::<i32>::from_shape_vec([2, 0], Vec::new()).unwrap();

    assert_eq!(scalar.nonzero().shape(), &[1, 0]);
    assert_eq!(scalar.argwhere().shape(), &[1, 0]);
    assert_eq!(empty.nonzero().shape(), &[0, 1]);
    assert_eq!(singleton.nonzero().data(), &[0]);
    assert_eq!(zero_length_axis.argwhere().shape(), &[0, 2]);
}

#[test]
fn take_handles_empty_indices_singletons_and_zero_length_axes() {
    let singleton = NDArray::from_shape_vec([1], vec![7_i32]).unwrap();
    let empty_axis = NDArray::<i32>::from_shape_vec([2, 0], Vec::new()).unwrap();
    let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();

    assert_eq!(singleton.take(&[-1], 0_i32).unwrap().data(), &[7]);
    let empty_taken = empty_axis.take(&[] as &[i32], 1_i32).unwrap();
    assert_eq!(empty_taken.shape(), &[2, 0]);
    assert!(empty_taken.data().is_empty());
    assert_eq!(
        scalar.take(&[] as &[i32], 0_i32).unwrap_err(),
        AtlasNdError::InvalidAxis { axis: 0, ndim: 0 }
    );
}

#[test]
fn put_and_scatter_handle_empty_singleton_zero_length_and_scalar_inputs() {
    let mut singleton = NDArray::from_shape_vec([1], vec![0_i32]).unwrap();
    let singleton_values = NDArray::from_shape_vec([1], vec![9_i32]).unwrap();
    let mut empty_axis = NDArray::<i32>::from_shape_vec([2, 0], Vec::new()).unwrap();
    let empty_values = NDArray::<i32>::from_shape_vec([2, 0], Vec::new()).unwrap();
    let mut scalar = NDArray::from_shape_vec([], vec![0_i32]).unwrap();

    singleton.scatter(&[0], &singleton_values, 0_i32).unwrap();
    empty_axis.put(&[] as &[i32], &empty_values, 1_i32).unwrap();

    assert_eq!(singleton.data(), &[9]);
    assert!(empty_axis.data().is_empty());
    assert_eq!(
        scalar
            .put(&[] as &[i32], &NDArray::from_shape_vec([], vec![1_i32]).unwrap(), 0_i32)
            .unwrap_err(),
        AtlasNdError::InvalidAxis { axis: 0, ndim: 0 }
    );
}
