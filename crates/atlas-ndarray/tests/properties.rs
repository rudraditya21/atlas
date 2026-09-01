mod support;

use atlas_ndarray::{AtlasNdError, NDArray};
use proptest::prelude::*;
use support::{
    array_strategy, assert_owned_invariants, assert_view_invariants, shape_strategy,
    view_fixture_strategy,
};

proptest! {
    #[test]
    fn generated_owned_arrays_preserve_invariants(array in array_strategy()) {
        assert_owned_invariants(&array);
    }

    #[test]
    fn generated_views_preserve_invariants_and_materialize_logically(
        fixture in view_fixture_strategy(),
    ) {
        let view = fixture.view();
        assert_view_invariants(&view);
    }

    #[test]
    fn allocation_constructors_preserve_generated_shape_invariants(
        shape in shape_strategy(),
        value in any::<i32>(),
    ) {
        let empty = NDArray::<i32>::empty(shape.clone()).unwrap();
        let full = NDArray::full(shape.clone(), value).unwrap();
        let zeros = NDArray::<i32>::zeros(shape.clone()).unwrap();
        let ones = NDArray::<i32>::ones(shape.clone()).unwrap();

        for array in [&empty, &full, &zeros, &ones] {
            prop_assert_eq!(array.shape(), shape.as_slice());
            prop_assert!(array.is_contiguous());
            assert_owned_invariants(array);
        }

        prop_assert!(empty.data().iter().all(|&element| element == 0));
        prop_assert!(full.data().iter().all(|&element| element == value));
        prop_assert!(zeros.data().iter().all(|&element| element == 0));
        prop_assert!(ones.data().iter().all(|&element| element == 1));
    }
}

#[test]
fn scalar_and_zero_sized_arrays_are_valid_fixture_inputs() {
    let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();
    let zero_sized = NDArray::<i32>::from_shape_vec([2, 0, 3], Vec::new()).unwrap();

    assert_owned_invariants(&scalar);
    assert_owned_invariants(&zero_sized);
    assert_view_invariants(&scalar.view());
    assert_view_invariants(&zero_sized.view());
}

#[test]
fn allocation_constructors_preserve_zero_length_axes() {
    let shape = [2, 0, 3];
    let empty = NDArray::<i32>::empty(shape).unwrap();
    let full = NDArray::full(shape, 7_i32).unwrap();
    let zeros = NDArray::<i32>::zeros(shape).unwrap();
    let ones = NDArray::<i32>::ones(shape).unwrap();

    for array in [&empty, &full, &zeros, &ones] {
        assert_eq!(array.shape(), shape);
        assert!(array.data().is_empty());
        assert_owned_invariants(array);
    }
}

#[test]
fn allocation_constructors_report_shape_overflow_consistently() {
    let expected = AtlasNdError::ShapeOverflow { op: "element count", shape: vec![usize::MAX, 2] };

    assert_eq!(NDArray::<i32>::empty([usize::MAX, 2]).unwrap_err(), expected);
    assert_eq!(NDArray::full([usize::MAX, 2], 7_i32).unwrap_err(), expected);
    assert_eq!(NDArray::<i32>::zeros([usize::MAX, 2]).unwrap_err(), expected);
    assert_eq!(NDArray::<i32>::ones([usize::MAX, 2]).unwrap_err(), expected);
}
