use atlas_ndarray::NDArray;
use atlas_random::{
    AtlasRng, bernoulli, categorical, choice, choice_indices, choice_indices_with_replacement,
    normal, permutation, rand, randint, randn, shuffle_axis, uniform,
};

#[test]
fn seeded_public_distribution_sequences_are_regression_stable() {
    let source = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let values = NDArray::from_shape_vec([5], vec![10_i32, 20, 30, 40, 50]).unwrap();
    let mut left = AtlasRng::seed_from_u64(9_271);
    let mut right = AtlasRng::seed_from_u64(9_271);

    assert_eq!(
        bernoulli([2, 3], 0.25, &mut left).unwrap().data(),
        bernoulli([2, 3], 0.25, &mut right).unwrap().data()
    );
    assert_eq!(
        categorical([4], &[1.0, 3.0, 2.0], &mut left).unwrap().data(),
        categorical([4], &[1.0, 3.0, 2.0], &mut right).unwrap().data()
    );
    assert_eq!(
        choice(&values, 3, &mut left).unwrap().data(),
        choice(&values, 3, &mut right).unwrap().data()
    );
    assert_eq!(
        choice_indices(5, 3, &mut left).unwrap().data(),
        choice_indices(5, 3, &mut right).unwrap().data()
    );
    assert_eq!(
        choice_indices_with_replacement(3, 5, &mut left).unwrap().data(),
        choice_indices_with_replacement(3, 5, &mut right).unwrap().data()
    );
    assert_eq!(
        normal([3], 1.0_f64, 2.0, &mut left).unwrap().data(),
        normal([3], 1.0_f64, 2.0, &mut right).unwrap().data()
    );
    assert_eq!(
        permutation(5, &mut left).unwrap().data(),
        permutation(5, &mut right).unwrap().data()
    );
    assert_eq!(rand([3], &mut left).unwrap().data(), rand([3], &mut right).unwrap().data());
    assert_eq!(
        randint([3], -2_i32, 4, &mut left).unwrap().data(),
        randint([3], -2_i32, 4, &mut right).unwrap().data()
    );
    assert_eq!(randn([3], &mut left).unwrap().data(), randn([3], &mut right).unwrap().data());
    assert_eq!(
        shuffle_axis(&source, 0, &mut left).unwrap().data(),
        shuffle_axis(&source, 0, &mut right).unwrap().data()
    );
    assert_eq!(
        uniform([3], -1.0_f64, 1.0, &mut left).unwrap().data(),
        uniform([3], -1.0_f64, 1.0, &mut right).unwrap().data()
    );
}

#[test]
fn seeded_sequential_and_parallel_fills_remain_independently_reproducible() {
    let mut sequential_left = AtlasRng::seed_from_u64(9_273);
    let mut sequential_right = AtlasRng::seed_from_u64(9_273);
    let mut parallel_left = AtlasRng::seed_from_u64(9_277);
    let mut parallel_right = AtlasRng::seed_from_u64(9_277);

    assert_eq!(
        uniform([64], -1.0_f64, 1.0, &mut sequential_left).unwrap().data(),
        uniform([64], -1.0_f64, 1.0, &mut sequential_right).unwrap().data()
    );
    assert_eq!(
        normal([64], 0.0_f64, 1.0, &mut sequential_left).unwrap().data(),
        normal([64], 0.0_f64, 1.0, &mut sequential_right).unwrap().data()
    );

    assert_eq!(
        bernoulli([1 << 20], 0.25, &mut parallel_left).unwrap().data(),
        bernoulli([1 << 20], 0.25, &mut parallel_right).unwrap().data()
    );
    assert_eq!(
        uniform([1 << 20], -1.0_f64, 1.0, &mut parallel_left).unwrap().data(),
        uniform([1 << 20], -1.0_f64, 1.0, &mut parallel_right).unwrap().data()
    );
    assert_eq!(
        normal([1 << 20], 0.0_f64, 1.0, &mut parallel_left).unwrap().data(),
        normal([1 << 20], 0.0_f64, 1.0, &mut parallel_right).unwrap().data()
    );
    assert_eq!(
        randint([4], 0_i32, 10, &mut parallel_left).unwrap().data(),
        randint([4], 0_i32, 10, &mut parallel_right).unwrap().data()
    );
}
