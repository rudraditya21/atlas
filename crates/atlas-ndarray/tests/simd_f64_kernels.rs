use atlas_ndarray::NDArray;

#[test]
fn contiguous_f64_unary_kernels_match_scalar_reference_with_special_values_and_a_tail() {
    let input = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0, -3.5, -1.0, 0.0, 1.0, 3.5];
    let values = NDArray::from_shape_vec([3, 3], input.to_vec()).unwrap();

    assert_f64_reference(values.neg().data(), &input.map(|value| -value));
    assert_f64_reference(values.abs().data(), &input.map(f64::abs));
}

#[test]
fn contiguous_f64_binary_kernels_match_scalar_reference_with_special_values_and_a_tail() {
    let lhs_values = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0, -3.5, -1.0, 0.0, 1.0, 3.5];
    let rhs_values = [2.0_f64; 9];
    let lhs = NDArray::from_shape_vec([3, 3], lhs_values.to_vec()).unwrap();
    let rhs = NDArray::from_shape_vec([3, 3], rhs_values.to_vec()).unwrap();

    assert_f64_reference(lhs.add(&rhs).unwrap().data(), &lhs_values.map(|value| value + 2.0_f64));
    assert_f64_reference(lhs.sub(&rhs).unwrap().data(), &lhs_values.map(|value| value - 2.0_f64));
    assert_f64_reference(lhs.mul(&rhs).unwrap().data(), &lhs_values.map(|value| value * 2.0_f64));
    assert_f64_reference(lhs.div(&rhs).unwrap().data(), &lhs_values.map(|value| value / 2.0_f64));
}

fn assert_f64_reference(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());

    for (&actual, &expected) in actual.iter().zip(expected) {
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else if expected.is_infinite() {
            assert_eq!(actual, expected);
        } else if expected == 0.0 {
            assert_eq!(actual, expected);
            assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
        } else {
            assert!((actual - expected).abs() <= expected.abs() * 1e-12 + 1e-12);
        }
    }
}
