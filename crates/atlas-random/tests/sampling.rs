use atlas_random::{AtlasRandomError, AtlasRng, normal, uniform};

#[test]
fn uniform_is_reproducible_for_seeded_rngs() {
    let mut left = AtlasRng::seed_from_u64(101);
    let mut right = AtlasRng::seed_from_u64(101);

    let lhs = uniform([3, 2], 5_i32, 15_i32, &mut left).unwrap();
    let rhs = uniform([3, 2], 5_i32, 15_i32, &mut right).unwrap();

    assert_eq!(lhs.shape(), &[3, 2]);
    assert_eq!(lhs.data(), rhs.data());
    assert!(lhs.data().iter().all(|value| *value >= 5 && *value < 15));
}

#[test]
fn normal_generates_ndarray_outputs_for_float_shapes() {
    let mut rng = AtlasRng::seed_from_u64(202);
    let sampled = normal([2, 2], 1.0_f64, 0.5, &mut rng).unwrap();

    assert_eq!(sampled.shape(), &[2, 2]);
    assert_eq!(sampled.len(), 4);
    assert!(sampled.data().iter().all(|value| value.is_finite()));
}

#[test]
fn invalid_sampling_arguments_return_structured_errors() {
    let mut rng = AtlasRng::seed_from_u64(303);

    assert!(matches!(
        uniform([1], 4_i32, 4_i32, &mut rng).unwrap_err(),
        AtlasRandomError::InvalidArgument { op: "uniform", .. }
    ));
    assert!(matches!(
        normal([1], 0.0_f64, -1.0, &mut rng).unwrap_err(),
        AtlasRandomError::InvalidArgument { op: "normal", .. }
    ));
}
