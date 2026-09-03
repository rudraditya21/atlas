use std::fmt::Debug;

use atlas_ndarray::{ArrayElement, DType, NDArray};

fn array<T: ArrayElement, const N: usize>(shape: [usize; N], values: Vec<T>) -> NDArray<T> {
    NDArray::from_shape_vec(shape, values).expect("fixture shape matches values")
}

fn assert_array<T: ArrayElement + PartialEq + Debug>(
    actual: &NDArray<T>,
    shape: &[usize],
    values: &[T],
) {
    assert_eq!(actual.shape(), shape);
    assert_eq!(actual.data(), values);
    assert!(actual.is_contiguous());
}

#[test]
fn unary_operations_preserve_logical_values_across_scalar_contiguous_strided_and_empty_inputs() {
    let scalar = array([], vec![-3_i32]);
    let contiguous = array([2, 2], vec![-3_i32, -1, 0, 2]);
    let strided = contiguous.view().transpose();
    let empty = NDArray::<i32>::zeros([2, 0]).unwrap();

    assert_array(&scalar.neg(), &[], &[3]);
    assert_array(&contiguous.abs(), &[2, 2], &[3, 1, 0, 2]);
    assert_array(&strided.sign(), &[2, 2], &[-1, 0, -1, 1]);
    assert_array(&empty.round(), &[2, 0], &[]);

    let floats = array([2, 2], vec![f64::NAN, f64::INFINITY, -1.5, 2.5]);
    let float_view = floats.view().transpose();
    let rounded = float_view.round();
    assert_eq!(rounded.shape(), &[2, 2]);
    assert!(rounded.data()[0].is_nan());
    assert_eq!(&rounded.data()[1..], &[-2.0, f64::INFINITY, 3.0]);
    assert!(float_view.isnan().data()[0]);
    assert!(float_view.isinf().data()[2]);
    assert_eq!(float_view.isfinite().data(), &[false, true, false, true]);
    assert_eq!(contiguous.isnan().data(), &[false, false, false, false]);
    assert_eq!(contiguous.isfinite().data(), &[true, true, true, true]);
}

#[test]
fn arithmetic_paths_match_manual_logical_results_and_promoted_dtypes() {
    let lhs = array([2, 1], vec![5_i32, 9]);
    let rhs = array([1, 2], vec![2_u8, 4]);
    let scalar = array([], vec![2_i32]);
    let empty = NDArray::<i32>::zeros([0, 2]).unwrap();

    let add = lhs.add(&rhs).unwrap();
    let sub = lhs.sub(&rhs).unwrap();
    let mul = lhs.mul(&rhs).unwrap();
    let div = lhs.div(&rhs).unwrap();
    let rem = lhs.rem(&rhs).unwrap();
    let minimum = lhs.minimum(&rhs).unwrap();
    let maximum = lhs.maximum(&rhs).unwrap();

    for result in [&add, &sub, &mul, &div, &rem, &minimum, &maximum] {
        assert_eq!(result.dtype(), DType::I32);
        assert_eq!(result.shape(), &[2, 2]);
    }

    assert_eq!(add.data(), &[7, 9, 11, 13]);
    assert_eq!(sub.data(), &[3, 1, 7, 5]);
    assert_eq!(mul.data(), &[10, 20, 18, 36]);
    assert_eq!(div.data(), &[2, 1, 4, 2]);
    assert_eq!(rem.data(), &[1, 1, 1, 1]);
    assert_eq!(minimum.data(), &[2, 4, 2, 4]);
    assert_eq!(maximum.data(), &[5, 5, 9, 9]);

    assert_array(&lhs.add(2_i32), &[2, 1], &[7, 11]);
    assert_array(&lhs.add(&scalar).unwrap(), &[2, 1], &[7, 11]);
    assert_array(&empty.rem(2_i32).unwrap(), &[0, 2], &[]);
    assert_array(&empty.maximum(&scalar).unwrap(), &[0, 2], &[]);

    let simd_lhs = array([8], (0..8).map(|value| value as f64).collect());
    let simd_rhs = array([8], vec![0.5_f64; 8]);
    assert_eq!(simd_lhs.add(&simd_rhs).unwrap().data(), &[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]);
    assert_eq!(simd_lhs.mul(&simd_rhs).unwrap().data(), &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]);
}

#[test]
fn ordering_logical_and_bitwise_operations_cover_scalar_broadcast_and_empty_paths() {
    let lhs = array([2, 1], vec![1_i32, 3]);
    let rhs = array([1, 2], vec![2_i32, 3]);
    assert_eq!(lhs.lt(&rhs).unwrap().data(), &[true, true, false, false]);
    assert_eq!(lhs.ge(&rhs).unwrap().data(), &[false, false, true, true]);

    let mask = array([2, 1], vec![true, false]);
    let other_mask = array([1, 2], vec![true, false]);
    assert_eq!(mask.logical_and(&other_mask).unwrap().data(), &[true, false, false, false]);
    assert_eq!(mask.logical_or(&other_mask).unwrap().data(), &[true, true, true, false]);
    assert_eq!(mask.logical_xor(true).data(), &[false, true]);
    assert_eq!(mask.logical_not().data(), &[false, true]);

    let bits = array([2, 1], vec![0b1010_u8, 0b1100]);
    let other_bits = array([1, 2], vec![0b0110_u8, 0b0011]);
    assert_eq!(bits.bitand(&other_bits).unwrap().data(), &[0b0010, 0b0010, 0b0100, 0]);
    assert_eq!(bits.bitor(0b0001_u8).data(), &[0b1011, 0b1101]);
    assert_eq!(bits.bitxor(0b1111_u8).data(), &[0b0101, 0b0011]);
    assert_eq!(bits.bitnot().data(), &[!0b1010_u8, !0b1100_u8]);

    let empty_mask = NDArray::from_shape_vec([0, 2], Vec::<bool>::new()).unwrap();
    let empty_bits = NDArray::<u8>::zeros([0, 2]).unwrap();
    assert_array(&empty_mask.logical_or(false), &[0, 2], &[]);
    assert_array(&empty_bits.bitand(0b1111_u8), &[0, 2], &[]);
}

#[test]
fn where_and_clip_preserve_promoted_logical_values_for_views_and_empty_inputs() {
    let condition_source = array([2, 2], vec![true, false, false, true]);
    let values_source = array([2, 2], vec![1_i32, 2, 3, 4]);
    let condition = condition_source.view().transpose();
    let values = values_source.view().transpose();
    let selected = condition.r#where(values, 0.5_f32).unwrap();

    assert_eq!(selected.dtype(), DType::F64);
    assert_array(&selected, &[2, 2], &[1.0_f64, 0.5, 0.5, 4.0]);

    let clipped = values_source.view().transpose().clip(1.5_f32, 3.5_f64).unwrap();
    assert_eq!(clipped.dtype(), DType::F64);
    assert_array(&clipped, &[2, 2], &[1.5_f64, 3.0, 2.0, 3.5]);

    let empty = NDArray::<i32>::zeros([0, 2]).unwrap();
    let empty_condition = NDArray::from_shape_vec([0, 2], Vec::<bool>::new()).unwrap();
    let selected_empty = empty_condition.r#where(&empty, 1.0_f32).unwrap();
    let clipped_empty = empty.clip(0.5_f32, 2.5_f64).unwrap();
    assert_eq!(selected_empty.dtype(), DType::F64);
    assert_eq!(clipped_empty.dtype(), DType::F64);
    assert_array(&selected_empty, &[0, 2], &[]);
    assert_array(&clipped_empty, &[0, 2], &[]);
}
