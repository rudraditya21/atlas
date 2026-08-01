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
fn mixed_seeded_sampling_is_repeatable_across_calls() {
    let mut left = AtlasRng::seed_from_u64(202);
    let mut right = AtlasRng::seed_from_u64(202);

    let left_uniform = uniform([2, 2], 0.0_f64, 1.0, &mut left).unwrap();
    let right_uniform = uniform([2, 2], 0.0_f64, 1.0, &mut right).unwrap();
    let left_normal = normal([2, 2], 1.0_f64, 0.5, &mut left).unwrap();
    let right_normal = normal([2, 2], 1.0_f64, 0.5, &mut right).unwrap();

    assert_eq!(left_uniform.shape(), &[2, 2]);
    assert_eq!(left_uniform.data(), right_uniform.data());
    assert_eq!(left_normal.shape(), &[2, 2]);
    assert_eq!(left_normal.data(), right_normal.data());
    assert!(left_normal.data().iter().all(|value| value.is_finite()));
}

#[test]
fn scalar_shapes_are_seeded_and_repeatable() {
    let mut left = AtlasRng::seed_from_u64(404);
    let mut right = AtlasRng::seed_from_u64(404);

    let left_uniform = uniform([], 10_i32, 20_i32, &mut left).unwrap();
    let right_uniform = uniform([], 10_i32, 20_i32, &mut right).unwrap();
    let left_normal = normal([], 0.0_f64, 1.0, &mut left).unwrap();
    let right_normal = normal([], 0.0_f64, 1.0, &mut right).unwrap();

    assert_eq!(left_uniform.shape(), &[] as &[usize]);
    assert_eq!(left_uniform.len(), 1);
    assert_eq!(left_uniform.data(), right_uniform.data());
    assert_eq!(left_normal.shape(), &[] as &[usize]);
    assert_eq!(left_normal.len(), 1);
    assert_eq!(left_normal.data(), right_normal.data());
}

#[test]
fn empty_shapes_return_empty_arrays_without_sampling_values() {
    let mut left = AtlasRng::seed_from_u64(505);
    let mut right = AtlasRng::seed_from_u64(505);

    let empty_uniform = uniform([0, 3], 0_i32, 10_i32, &mut left).unwrap();
    let empty_normal = normal([0], 0.0_f64, 1.0, &mut left).unwrap();
    let post_empty_uniform = uniform([2], 0_i32, 10_i32, &mut left).unwrap();
    let direct_uniform = uniform([2], 0_i32, 10_i32, &mut right).unwrap();

    assert_eq!(empty_uniform.shape(), &[0, 3]);
    assert_eq!(empty_uniform.len(), 0);
    assert!(empty_uniform.data().is_empty());
    assert_eq!(empty_normal.shape(), &[0]);
    assert_eq!(empty_normal.len(), 0);
    assert!(empty_normal.data().is_empty());
    assert_eq!(post_empty_uniform.data(), direct_uniform.data());
}

#[test]
fn invalid_sampling_arguments_return_exact_errors() {
    let mut rng = AtlasRng::seed_from_u64(303);

    assert_eq!(
        uniform([1], 4_i32, 4_i32, &mut rng).unwrap_err(),
        AtlasRandomError::InvalidArgument {
            op: "uniform",
            reason: "low must be strictly less than high",
        }
    );
    assert_eq!(
        normal([1], 0.0_f64, -1.0, &mut rng).unwrap_err(),
        AtlasRandomError::InvalidArgument {
            op: "normal",
            reason: "stddev must be strictly positive",
        }
    );
}
