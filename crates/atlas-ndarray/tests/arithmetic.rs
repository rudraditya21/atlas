use atlas_ndarray::{AtlasNdError, DType, NDArray, Numeric};

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
    assert_eq!(array.div(2).unwrap().data(), &[0, 1, 1]);
}

#[test]
fn scalar_values_match_scalar_shaped_array_results() {
    let array = NDArray::from_vec([2, 2], vec![2_i32, 4, 6, 8]).unwrap();
    let scalar = NDArray::from_shape_vec([], vec![2_i32]).unwrap();

    assert_array_eq(&array.add(2), &array.add(&scalar).unwrap());
    assert_array_eq(&array.sub(2), &array.sub(&scalar).unwrap());
    assert_array_eq(&array.mul(2), &array.mul(&scalar).unwrap());
    assert_array_eq(&array.div(2).unwrap(), &array.div(&scalar).unwrap());
}

#[test]
fn scalar_and_scalar_shaped_array_paths_match_for_empty_outputs() {
    let empty = NDArray::<i32>::zeros([0, 3]).unwrap();
    let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();

    assert_array_eq(&empty.add(7), &empty.add(&scalar).unwrap());
    assert_array_eq(&empty.sub(7), &empty.sub(&scalar).unwrap());
    assert_array_eq(&empty.mul(7), &empty.mul(&scalar).unwrap());
    assert_array_eq(&empty.div(7).unwrap(), &empty.div(&scalar).unwrap());
}

#[test]
fn scalar_arithmetic_fast_paths_preserve_f32_special_values_and_tails() {
    let values = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let array = NDArray::from_shape_vec([3, 3], values.to_vec()).unwrap();

    assert_f32_results(array.add(1.0_f32).data(), &values.map(|value| value + 1.0_f32));
    assert_f32_results(array.sub(1.0_f32).data(), &values.map(|value| value - 1.0_f32));
    assert_f32_results(array.mul(f32::INFINITY).data(), &values.map(|value| value * f32::INFINITY));
    assert_f32_results(array.div(0.0_f32).unwrap().data(), &values.map(|value| value / 0.0_f32));
}

#[test]
fn scalar_arithmetic_fast_paths_preserve_f64_special_values_and_tails() {
    let values = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -2.0, -1.0];
    let array = NDArray::from_shape_vec([5], values.to_vec()).unwrap();

    assert_f64_results(array.add(1.0).data(), &values.map(|value| value + 1.0));
    assert_f64_results(array.sub(1.0).data(), &values.map(|value| value - 1.0));
    assert_f64_results(array.mul(f64::INFINITY).data(), &values.map(|value| value * f64::INFINITY));
    assert_f64_results(array.div(0.0).unwrap().data(), &values.map(|value| value / 0.0));
}

#[test]
fn scalar_arithmetic_fast_paths_preserve_empty_arrays() {
    let f32_values = NDArray::<f32>::zeros([0, 3]).unwrap();
    let f64_values = NDArray::<f64>::zeros([0]).unwrap();

    assert!(f32_values.add(1.0).is_empty());
    assert!(f32_values.sub(1.0).is_empty());
    assert!(f32_values.mul(2.0).is_empty());
    assert!(f32_values.div(2.0).unwrap().is_empty());
    assert!(f64_values.add(1.0).is_empty());
    assert!(f64_values.sub(1.0).is_empty());
    assert!(f64_values.mul(2.0).is_empty());
    assert!(f64_values.div(2.0).unwrap().is_empty());
}

fn assert_f32_results(actual: &[f32], expected: &[f32]) {
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert_eq!(actual.is_nan(), expected.is_nan());
        assert_eq!(actual.is_infinite(), expected.is_infinite());
        assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
        if actual.is_finite() {
            assert_eq!(actual, expected);
        }
    }
}

fn assert_f64_results(actual: &[f64], expected: &[f64]) {
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert_eq!(actual.is_nan(), expected.is_nan());
        assert_eq!(actual.is_infinite(), expected.is_infinite());
        assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
        if actual.is_finite() {
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn mixed_scalar_arithmetic_promotes_results_before_dispatch() {
    let array = NDArray::from_vec([2], vec![2_i32, 4]).unwrap();

    let added = array.add(3_u32);
    let divided = array.div(2_f32).unwrap();

    assert_eq!(added.dtype(), DType::I64);
    assert_eq!(added.data(), &[5_i64, 7]);

    assert_eq!(divided.dtype(), DType::F64);
    assert_eq!(divided.data(), &[1.0_f64, 2.0]);
}

#[test]
fn mixed_array_arithmetic_promotes_dtype_through_broadcast_paths() {
    let ints = NDArray::from_vec([2, 1], vec![1_i32, 2]).unwrap();
    let floats = NDArray::from_vec([1, 2], vec![0.5_f32, 1.5]).unwrap();

    let added = ints.add(&floats).unwrap();
    let multiplied = ints.mul(&floats).unwrap();

    assert_eq!(added.dtype(), DType::F64);
    assert_eq!(added.data(), &[1.5_f64, 2.5, 2.5, 3.5]);

    assert_eq!(multiplied.dtype(), DType::F64);
    assert_eq!(multiplied.data(), &[0.5_f64, 1.5, 1.0, 3.0]);
}

#[test]
fn mixed_integer_arrays_use_promotion_fallbacks_for_nonrepresentable_ranges() {
    let signed = NDArray::from_vec([2], vec![1_i64, 2]).unwrap();
    let unsigned = NDArray::from_vec([2], vec![3_u64, 4]).unwrap();

    let result = signed.add(&unsigned).unwrap();

    assert_eq!(result.dtype(), DType::F64);
    assert_eq!(result.data(), &[4.0_f64, 6.0]);
}
