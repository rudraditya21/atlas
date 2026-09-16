use atlas_ndarray::NDArray;

#[test]
fn contiguous_f32_unary_kernels_match_scalar_reference_with_a_tail() {
    let input = [-4.0_f32, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    let values = NDArray::from_shape_vec([3, 3], input.to_vec()).unwrap();

    assert_eq!(values.neg().data(), &input.map(|value| -value));
    assert_eq!(values.abs().data(), &input.map(f32::abs));
}

#[test]
fn contiguous_f32_binary_kernels_match_scalar_reference_with_a_tail() {
    let lhs_values = [-4.0_f32, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    let rhs_values = [2.0_f32; 9];
    let lhs = NDArray::from_shape_vec([3, 3], lhs_values.to_vec()).unwrap();
    let rhs = NDArray::from_shape_vec([3, 3], rhs_values.to_vec()).unwrap();

    assert_eq!(lhs.add(&rhs).unwrap().data(), &lhs_values.map(|value| value + 2.0_f32));
    assert_eq!(lhs.sub(&rhs).unwrap().data(), &lhs_values.map(|value| value - 2.0_f32));
    assert_eq!(lhs.mul(&rhs).unwrap().data(), &lhs_values.map(|value| value * 2.0_f32));
    assert_eq!(lhs.div(&rhs).unwrap().data(), &lhs_values.map(|value| value / 2.0_f32));
}
