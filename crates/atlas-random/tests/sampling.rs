use atlas_ndarray::{AtlasNdError, NDArray, compute_strides, element_count};
use atlas_random::{AtlasRandomError, AtlasRng, normal, rand, randn, uniform};

fn assert_layout_invariants<T>(array: &NDArray<T>, expected_shape: &[usize])
where
    T: atlas_ndarray::Numeric,
{
    assert_eq!(array.shape(), expected_shape);
    assert_eq!(array.strides(), compute_strides(expected_shape));
    assert_eq!(array.len(), element_count(expected_shape));
    assert_eq!(array.ndim(), expected_shape.len());
    assert!(array.is_contiguous());
    assert_eq!(array.is_empty(), expected_shape.iter().product::<usize>() == 0);
}

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
fn uniform_outputs_satisfy_bounds_and_layout_invariants() {
    let mut rng = AtlasRng::seed_from_u64(909);

    let matrix = uniform([2, 3], -3.0_f64, 2.0, &mut rng).unwrap();
    let scalar = uniform([], 10_i32, 20_i32, &mut rng).unwrap();
    let empty = uniform([0, 2], 0_i32, 10_i32, &mut rng).unwrap();

    assert_layout_invariants(&matrix, &[2, 3]);
    assert!(matrix.data().iter().all(|value| *value >= -3.0 && *value < 2.0));
    assert_layout_invariants(&scalar, &[] as &[usize]);
    assert!(scalar.data()[0] >= 10 && scalar.data()[0] < 20);
    assert_layout_invariants(&empty, &[0, 2]);
    assert!(empty.data().is_empty());
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
fn normal_outputs_are_finite_and_layout_correct() {
    let mut rng = AtlasRng::seed_from_u64(1001);

    let matrix = normal([2, 2], 1.0_f64, 0.5, &mut rng).unwrap();
    let tensor = normal([2, 1, 2], 0.0_f64, 1.0, &mut rng).unwrap();
    let scalar = normal([], 0.0_f64, 1.0, &mut rng).unwrap();
    let empty = normal([0], 0.0_f64, 1.0, &mut rng).unwrap();

    assert_layout_invariants(&matrix, &[2, 2]);
    assert!(matrix.data().iter().all(|value| value.is_finite()));
    assert_layout_invariants(&tensor, &[2, 1, 2]);
    assert!(tensor.data().iter().all(|value| value.is_finite()));
    assert_layout_invariants(&scalar, &[] as &[usize]);
    assert!(scalar.data()[0].is_finite());
    assert_layout_invariants(&empty, &[0]);
    assert!(empty.data().is_empty());
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
fn sampling_preserves_requested_shapes() {
    let mut rng = AtlasRng::seed_from_u64(606);

    let uniform_matrix = uniform([2, 3], 0_i32, 10_i32, &mut rng).unwrap();
    let normal_tensor = normal([2, 1, 2], 0.0_f64, 1.0, &mut rng).unwrap();
    let uniform_scalar = uniform([], 0_i32, 10_i32, &mut rng).unwrap();
    let normal_empty = normal([0, 2], 0.0_f64, 1.0, &mut rng).unwrap();

    assert_layout_invariants(&uniform_matrix, &[2, 3]);
    assert_layout_invariants(&normal_tensor, &[2, 1, 2]);
    assert_layout_invariants(&uniform_scalar, &[] as &[usize]);
    assert_layout_invariants(&normal_empty, &[0, 2]);
}

#[test]
fn rand_and_randn_cover_numpy_style_default_float_sampling() {
    let mut left = AtlasRng::seed_from_u64(7070);
    let mut right = AtlasRng::seed_from_u64(7070);

    let left_rand = rand([2, 3], &mut left).unwrap();
    let right_rand = rand([2, 3], &mut right).unwrap();
    let left_randn = randn([], &mut left).unwrap();
    let right_randn = randn([], &mut right).unwrap();

    assert_layout_invariants(&left_rand, &[2, 3]);
    assert!(left_rand.data().iter().all(|value| *value >= 0.0 && *value < 1.0));
    assert_eq!(left_rand.data(), right_rand.data());
    assert_layout_invariants(&left_randn, &[] as &[usize]);
    assert!(left_randn.data()[0].is_finite());
    assert_eq!(left_randn.data(), right_randn.data());
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
fn uniform_rejects_invalid_bounds_with_exact_errors() {
    let mut rng = AtlasRng::seed_from_u64(303);

    assert_eq!(
        uniform([1], 4_i32, 4_i32, &mut rng).unwrap_err(),
        AtlasRandomError::InvalidArgument {
            op: "uniform",
            reason: "low must be strictly less than high",
        }
    );
    assert_eq!(
        uniform([1], 5_i32, 4_i32, &mut rng).unwrap_err(),
        AtlasRandomError::InvalidArgument {
            op: "uniform",
            reason: "low must be strictly less than high",
        }
    );
    assert_eq!(
        uniform([1], f64::NAN, 1.0_f64, &mut rng).unwrap_err(),
        AtlasRandomError::InvalidArgument { op: "uniform", reason: "low and high must be finite" }
    );
    assert_eq!(
        uniform([1], 0.0_f64, f64::NAN, &mut rng).unwrap_err(),
        AtlasRandomError::InvalidArgument { op: "uniform", reason: "low and high must be finite" }
    );
    assert_eq!(
        uniform([1], f64::NEG_INFINITY, 1.0_f64, &mut rng).unwrap_err(),
        AtlasRandomError::InvalidArgument { op: "uniform", reason: "low and high must be finite" }
    );
    assert_eq!(
        uniform([1], 0.0_f64, f64::INFINITY, &mut rng).unwrap_err(),
        AtlasRandomError::InvalidArgument { op: "uniform", reason: "low and high must be finite" }
    );
}

#[test]
fn normal_rejects_invalid_stddev_with_exact_errors() {
    let mut rng = AtlasRng::seed_from_u64(707);

    assert_eq!(
        normal([1], 0.0_f64, 0.0, &mut rng).unwrap_err(),
        AtlasRandomError::InvalidArgument {
            op: "normal",
            reason: "stddev must be strictly positive",
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

#[test]
fn normal_rejects_non_finite_parameters_with_exact_errors() {
    let mut rng = AtlasRng::seed_from_u64(808);

    assert_eq!(
        normal([1], f64::NAN, 1.0, &mut rng).unwrap_err(),
        AtlasRandomError::InvalidArgument {
            op: "normal",
            reason: "mean and stddev must be finite",
        }
    );
    assert_eq!(
        normal([1], 0.0_f64, f64::NAN, &mut rng).unwrap_err(),
        AtlasRandomError::InvalidArgument {
            op: "normal",
            reason: "mean and stddev must be finite",
        }
    );
    assert_eq!(
        normal([1], f64::INFINITY, 1.0, &mut rng).unwrap_err(),
        AtlasRandomError::InvalidArgument {
            op: "normal",
            reason: "mean and stddev must be finite",
        }
    );
    assert_eq!(
        normal([1], 0.0_f64, f64::NEG_INFINITY, &mut rng).unwrap_err(),
        AtlasRandomError::InvalidArgument {
            op: "normal",
            reason: "mean and stddev must be finite",
        }
    );
}

#[test]
fn sampling_rejects_overflowing_shapes_with_exact_errors() {
    let mut rng = AtlasRng::seed_from_u64(909);

    assert_eq!(
        uniform([usize::MAX, 2], 0_i32, 10_i32, &mut rng).unwrap_err(),
        AtlasRandomError::NdArray(AtlasNdError::ShapeOverflow {
            op: "element count",
            shape: vec![usize::MAX, 2],
        })
    );
    assert_eq!(
        normal([usize::MAX, 2], 0.0_f64, 1.0, &mut rng).unwrap_err(),
        AtlasRandomError::NdArray(AtlasNdError::ShapeOverflow {
            op: "element count",
            shape: vec![usize::MAX, 2],
        })
    );
}
