use atlas_ndarray::{DType, NDArray};

#[test]
fn scalar_comparisons_return_bool_arrays_with_preserved_shape() {
    let array = NDArray::from_vec([2, 2], vec![1_i32, 2, 3, 4]).unwrap();

    let eq = array.eq(2);
    let gt = array.gt(2);

    assert_eq!(eq.dtype(), DType::Bool);
    assert_eq!(eq.shape(), &[2, 2]);
    assert_eq!(eq.data(), &[false, true, false, false]);

    assert_eq!(gt.dtype(), DType::Bool);
    assert_eq!(gt.data(), &[false, false, true, true]);
}

#[test]
fn broadcasted_array_comparisons_materialize_logical_bool_results() {
    let lhs = NDArray::from_vec([2, 1], vec![1_i32, 3]).unwrap();
    let rhs = NDArray::from_vec([1, 3], vec![2_i32, 3, 4]).unwrap();

    let lt = lhs.lt(&rhs).unwrap();
    let ge = lhs.ge(&rhs).unwrap();

    assert_eq!(lt.dtype(), DType::Bool);
    assert_eq!(lt.shape(), &[2, 3]);
    assert_eq!(lt.data(), &[true, true, true, false, false, true]);

    assert_eq!(ge.dtype(), DType::Bool);
    assert_eq!(ge.data(), &[false, false, false, true, true, false]);
}

#[test]
fn scalar_shaped_rhs_and_empty_outputs_follow_broadcast_comparison_rules() {
    let empty = NDArray::<i32>::zeros([0, 3]).unwrap();
    let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();

    let eq = empty.eq(&scalar).unwrap();

    assert_eq!(eq.dtype(), DType::Bool);
    assert_eq!(eq.shape(), &[0, 3]);
    assert!(eq.data().is_empty());
}

#[test]
fn bool_arrays_are_valid_comparison_outputs_and_support_indexing() {
    let lhs = NDArray::from_shape_vec([2], vec![true, false]).unwrap();
    let rhs = NDArray::from_shape_vec([2], vec![true, true]).unwrap();

    let eq = lhs.eq(&rhs).unwrap();
    let ne = lhs.ne(&rhs).unwrap();

    assert_eq!(eq.dtype(), DType::Bool);
    assert_eq!(eq.data(), &[true, false]);
    assert_eq!(ne.data(), &[false, true]);
    assert!(*eq.get(&[0]).unwrap());
    assert!(*ne.get(&[1]).unwrap());
}

#[test]
fn float_nan_comparisons_follow_ieee_predicates() {
    let values = NDArray::from_shape_vec([3], vec![f64::NAN, 1.0, f64::NAN]).unwrap();

    assert_eq!(values.eq(f64::NAN).data(), &[false; 3]);
    assert_eq!(values.ne(f64::NAN).data(), &[true; 3]);
    assert_eq!(values.lt(f64::NAN).data(), &[false; 3]);
    assert_eq!(values.le(f64::NAN).data(), &[false; 3]);
    assert_eq!(values.gt(f64::NAN).data(), &[false; 3]);
    assert_eq!(values.ge(f64::NAN).data(), &[false; 3]);
}
