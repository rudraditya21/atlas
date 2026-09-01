use atlas_ndarray::{ArrayElement, AtlasNdError, CastMode, DType, NDArray};

fn assert_owned_invariants<T: ArrayElement>(array: &NDArray<T>) {
    assert_eq!(array.data().len(), array.shape().iter().product());
    assert_eq!(array.shape().len(), array.strides().len());
    assert!(array.is_owned());
    assert!(array.is_contiguous());
}

macro_rules! identity_cast_test {
    ($name:ident, $type:ty, [$($value:expr),+ $(,)?]) => {
        #[test]
        fn $name() {
            let values: Vec<$type> = vec![$($value),+];
            let array = NDArray::from_shape_vec([values.len()], values.clone()).unwrap();
            let casted = array.astype::<$type>().unwrap();

            assert_eq!(casted.dtype(), DType::of::<$type>());
            assert_eq!(casted.data(), values.as_slice());
            assert_owned_invariants(&casted);
        }
    };
}

identity_cast_test!(casts_bool_values, bool, [false, true]);
identity_cast_test!(casts_i8_values, i8, [-1, 1]);
identity_cast_test!(casts_i16_values, i16, [-1, 1]);
identity_cast_test!(casts_i32_values, i32, [-1, 1]);
identity_cast_test!(casts_i64_values, i64, [-1, 1]);
identity_cast_test!(casts_isize_values, isize, [-1, 1]);
identity_cast_test!(casts_u8_values, u8, [0, 1]);
identity_cast_test!(casts_u16_values, u16, [0, 1]);
identity_cast_test!(casts_u32_values, u32, [0, 1]);
identity_cast_test!(casts_u64_values, u64, [0, 1]);
identity_cast_test!(casts_usize_values, usize, [0, 1]);
identity_cast_test!(casts_f32_values, f32, [-1.5, 1.5]);
identity_cast_test!(casts_f64_values, f64, [-1.5, 1.5]);

#[test]
fn checked_casts_reject_lossy_dtype_conversions() {
    let floats = NDArray::from_shape_vec([2], vec![1.0_f64, 2.0]).unwrap();

    assert_eq!(
        floats.astype::<f32>().unwrap_err(),
        AtlasNdError::InvalidCast { from: DType::F64, to: DType::F32, mode: CastMode::Checked }
    );
    assert_eq!(
        floats.astype::<i32>().unwrap_err(),
        AtlasNdError::InvalidCast { from: DType::F64, to: DType::I32, mode: CastMode::Checked }
    );
}

#[test]
fn lossy_casts_require_explicit_mode_and_preserve_output_invariants() {
    let floats = NDArray::from_shape_vec([2, 2], vec![1.25_f64, -2.75, 3.0, 4.5]).unwrap();
    let casted = floats.astype_with_mode::<i32>(CastMode::Lossy).unwrap();

    assert_eq!(casted.dtype(), DType::I32);
    assert_eq!(casted.shape(), floats.shape());
    assert_eq!(casted.data(), &[1, -2, 3, 4]);
    assert_owned_invariants(&casted);
}

#[test]
fn view_conversions_materialize_logical_values_and_asarray_borrows() {
    let array = NDArray::from_shape_vec([2, 3], (0_i32..6).collect()).unwrap();
    let view = array.view().transpose();
    let asarray = view.asarray();
    let casted = view.astype::<f64>().unwrap();

    assert!(asarray.is_borrowed());
    assert_eq!(asarray.view().data().as_ptr(), array.data().as_ptr());
    assert_eq!(casted.shape(), &[3, 2]);
    assert_eq!(casted.strides(), &[2, 1]);
    assert_eq!(casted.data(), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    assert_owned_invariants(&casted);
}

#[test]
fn lossy_float_casts_preserve_nan_and_infinity_but_reject_integer_targets() {
    let specials =
        NDArray::from_shape_vec([3], vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY]).unwrap();
    let floats = specials.astype_with_mode::<f32>(CastMode::Lossy).unwrap();

    assert!(floats.data()[0].is_nan());
    assert_eq!(floats.data()[1], f32::INFINITY);
    assert_eq!(floats.data()[2], f32::NEG_INFINITY);
    assert_eq!(
        specials.astype_with_mode::<i32>(CastMode::Lossy).unwrap_err(),
        AtlasNdError::InvalidCast { from: DType::F64, to: DType::I32, mode: CastMode::Lossy }
    );
}
