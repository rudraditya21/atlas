use atlas_ndarray::{AtlasNdError, NDArray};
use atlas_random::{
    AtlasRandomError, AtlasRng, bernoulli, categorical, permutation, randint, shuffle_axis,
};

#[test]
fn seeded_extended_random_apis_are_repeatable_and_shape_correct() {
    let mut left = AtlasRng::seed_from_u64(71);
    let mut right = AtlasRng::seed_from_u64(71);
    let left_bernoulli = bernoulli([2, 3], 0.25, &mut left).unwrap();
    let right_bernoulli = bernoulli([2, 3], 0.25, &mut right).unwrap();
    let left_integers = randint([4], -2_i32, 3, &mut left).unwrap();
    let right_integers = randint([4], -2_i32, 3, &mut right).unwrap();
    let left_categories = categorical([4], &[1.0, 3.0], &mut left).unwrap();
    let right_categories = categorical([4], &[1.0, 3.0], &mut right).unwrap();

    assert_eq!(left_bernoulli.shape(), &[2, 3]);
    assert_eq!(left_bernoulli.data(), right_bernoulli.data());
    assert_eq!(left_integers.data(), right_integers.data());
    assert!(left_integers.data().iter().all(|value| (-2..3).contains(value)));
    assert_eq!(left_categories.data(), right_categories.data());
    assert!(left_categories.data().iter().all(|value| *value < 2));
}

#[test]
fn permutation_and_axis_shuffle_are_lossless() {
    let values = NDArray::from_shape_vec([3, 2], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let mut rng = AtlasRng::seed_from_u64(73);
    let permutation = permutation(6, &mut rng).unwrap();
    let shuffled = shuffle_axis(&values, 0, &mut rng).unwrap();
    let mut permutation_values = permutation.data().to_vec();
    let mut shuffled_values = shuffled.data().to_vec();
    permutation_values.sort_unstable();
    shuffled_values.sort_unstable();

    assert_eq!(permutation_values, (0..6).collect::<Vec<_>>());
    assert_eq!(shuffled.shape(), values.shape());
    assert_eq!(shuffled_values, values.data());
}

#[test]
fn extended_random_apis_reject_invalid_parameters() {
    let values = NDArray::from_shape_vec([2, 2], vec![0_i32, 1, 2, 3]).unwrap();
    let mut rng = AtlasRng::seed_from_u64(79);

    assert!(matches!(
        bernoulli([1], -0.1, &mut rng).unwrap_err(),
        AtlasRandomError::InvalidArgument { op: "bernoulli", .. }
    ));
    assert!(matches!(
        categorical([1], &[0.0, 0.0], &mut rng).unwrap_err(),
        AtlasRandomError::InvalidArgument { op: "categorical", .. }
    ));
    assert_eq!(
        shuffle_axis(&values, 2, &mut rng).unwrap_err(),
        AtlasRandomError::NdArray(AtlasNdError::InvalidAxis { axis: 2, ndim: 2 })
    );
}
