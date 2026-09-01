mod support;

use atlas_ndarray::NDArray;
use proptest::prelude::*;
use support::{
    array_strategy, assert_owned_invariants, assert_view_invariants, view_fixture_strategy,
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
