use atlas_linalg::{AtlasLinalgError, cholesky, dot, matmul, qr};
use atlas_ndarray::{AtlasNdError, NDArray, checked_compute_strides, checked_element_count};
use atlas_random::{AtlasRandomError, AtlasRng, normal, uniform};
use atlas_stats::{AtlasStatsError, correlation, covariance, stddev, variance};

fn assert_close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() <= 1e-10);
}

#[test]
fn random_ndarray_stats_pipeline_preserves_variance_across_views() {
    let mut rng = AtlasRng::seed_from_u64(1_024);
    let sampled = uniform([2, 3], -1.0_f64, 1.0, &mut rng).unwrap();
    let transposed = sampled.view().transpose();

    let owned_variance = variance(&sampled).unwrap();
    let viewed_variance = variance(transposed.clone()).unwrap();
    let owned_stddev = stddev(&sampled).unwrap();
    let viewed_stddev = stddev(transposed).unwrap();

    assert_eq!(sampled.shape(), &[2, 3]);
    assert!(sampled.is_contiguous());
    assert!(sampled.data().iter().all(|value| *value >= -1.0 && *value < 1.0));
    assert_close(owned_variance, viewed_variance);
    assert_close(owned_stddev, viewed_stddev);
}

#[test]
fn ndarray_linalg_stats_pipeline_supports_vector_statistics() {
    let matrix = NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let weights = NDArray::from_shape_vec([2], vec![10.0_f64, 20.0]).unwrap();
    let expected = NDArray::from_shape_vec([3], vec![3.0_f64, 4.0, 5.0]).unwrap();

    let projected = matmul(matrix.view().transpose(), &weights).unwrap();

    assert_eq!(projected.shape(), &[3]);
    assert_close(covariance(&projected, &expected).unwrap(), 20.0);
    assert_close(correlation(&projected, &expected).unwrap(), 1.0);
}

#[test]
fn random_linalg_stats_pipeline_produces_finite_statistical_outputs() {
    let mut rng = AtlasRng::seed_from_u64(2_048);
    let lhs = normal([2, 3], 0.0_f64, 1.0, &mut rng).unwrap();
    let rhs = normal([3, 2], 0.0_f64, 1.0, &mut rng).unwrap();

    let product = matmul(&lhs, &rhs).unwrap();
    let product_variance = variance(&product).unwrap();
    let product_stddev = stddev(product.view().transpose()).unwrap();

    assert_eq!(product.shape(), &[2, 2]);
    assert!(product.is_contiguous());
    assert!(product.data().iter().all(|value| value.is_finite()));
    assert!(product_variance.is_finite());
    assert!(product_variance >= 0.0);
    assert!(product_stddev.is_finite());
    assert!(product_stddev >= 0.0);
}

#[test]
fn ndarray_qr_stats_pipeline_preserves_reconstruction_statistics() {
    let matrix = NDArray::from_shape_vec([3, 2], vec![1.0_f64, 1.0, 1.0, 0.0, 0.0, 1.0]).unwrap();

    let factors = qr(&matrix).unwrap();
    let reconstructed = matmul(&factors.q, &factors.r).unwrap();

    assert_eq!(factors.q.shape(), &[3, 2]);
    assert_eq!(factors.r.shape(), &[2, 2]);
    assert_eq!(reconstructed.shape(), matrix.shape());
    assert!(reconstructed.data().iter().all(|value| value.is_finite()));
    assert_close(variance(&reconstructed).unwrap(), variance(&matrix).unwrap());
    assert_close(stddev(reconstructed.view().transpose()).unwrap(), stddev(&matrix).unwrap());
}

#[test]
fn ndarray_cholesky_stats_pipeline_preserves_reconstruction_statistics() {
    let matrix = NDArray::from_shape_vec([2, 2], vec![4.0_f64, 2.0, 2.0, 3.0]).unwrap();

    let factor = cholesky(&matrix).unwrap();
    let reconstructed = matmul(&factor.l, factor.l.view().transpose()).unwrap();

    assert_eq!(factor.l.shape(), &[2, 2]);
    assert_eq!(reconstructed.shape(), matrix.shape());
    assert!(reconstructed.data().iter().all(|value| value.is_finite()));
    assert_close(variance(&reconstructed).unwrap(), variance(&matrix).unwrap());
    assert_close(stddev(reconstructed.view()).unwrap(), stddev(&matrix).unwrap());
}

#[test]
fn random_pipeline_reports_exact_boundary_errors() {
    let mut rng = AtlasRng::seed_from_u64(4_096);

    assert_eq!(
        normal([2], 0.0_f64, 0.0, &mut rng).unwrap_err(),
        AtlasRandomError::InvalidArgument {
            op: "normal",
            reason: "stddev must be strictly positive",
        }
    );
    assert_eq!(
        uniform([2], 1.0_f64, 1.0, &mut rng).unwrap_err(),
        AtlasRandomError::InvalidArgument {
            op: "uniform",
            reason: "low must be strictly less than high",
        }
    );
}

#[test]
fn ndarray_linalg_pipeline_reports_exact_boundary_errors() {
    let lhs = NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs = NDArray::from_shape_vec([2, 2], vec![7.0_f64, 8.0, 9.0, 10.0]).unwrap();
    let tensor =
        NDArray::from_shape_vec([1, 2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    assert_eq!(
        matmul(&lhs, &rhs).unwrap_err(),
        AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: vec![2, 3],
            right: vec![2, 2],
            reason: "left matrix column count must match right matrix row count",
        }
    );
    assert_eq!(
        matmul(&tensor, &rhs).unwrap_err(),
        AtlasLinalgError::InvalidOperandRank { op: "matmul", left: 3, right: 2 }
    );
}

#[test]
fn linalg_stats_pipeline_reports_exact_stats_boundary_errors() {
    let lhs = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 0.0, 1.0, 0.0]).unwrap();
    let rhs = NDArray::from_shape_vec([2], vec![5.0_f64, 5.0]).unwrap();
    let expected = NDArray::from_shape_vec([2], vec![1.0_f64, 2.0]).unwrap();
    let mismatched = NDArray::from_shape_vec([1], vec![3.0_f64]).unwrap();

    let projected = matmul(&lhs, &rhs).unwrap();

    assert_eq!(
        correlation(&projected, &expected).unwrap_err(),
        AtlasStatsError::ZeroVariance { op: "correlation" }
    );
    assert_eq!(
        covariance(&projected, &mismatched).unwrap_err(),
        AtlasStatsError::ShapeMismatch {
            op: "covariance",
            left: vec![2],
            right: vec![1],
            reason: "vector lengths must match",
        }
    );
}

#[test]
fn empty_vector_views_report_stable_stats_errors_across_crates() {
    let lhs_base = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let rhs_base = NDArray::from_shape_vec([3], vec![4.0_f64, 5.0, 6.0]).unwrap();
    let lhs = lhs_base.view().slice([3], [0]).unwrap();
    let rhs = rhs_base.view().slice([3], [0]).unwrap();

    assert_eq!(variance(lhs.clone()).unwrap_err(), AtlasStatsError::EmptyInput { op: "variance" });
    assert_eq!(stddev(lhs.clone()).unwrap_err(), AtlasStatsError::EmptyInput { op: "stddev" });
    assert_eq!(
        covariance(lhs.clone(), rhs.clone()).unwrap_err(),
        AtlasStatsError::EmptyInput { op: "covariance" }
    );
    assert_eq!(
        correlation(lhs, rhs).unwrap_err(),
        AtlasStatsError::EmptyInput { op: "correlation" }
    );
}

#[test]
fn empty_view_operands_remain_well_defined_for_linalg_dispatch() {
    let vector_base = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let empty_vector = vector_base.view().slice([3], [0]).unwrap();

    assert_eq!(dot(empty_vector.clone(), empty_vector.clone()).unwrap(), 0.0);

    let vector_product = matmul(empty_vector.clone(), empty_vector.clone()).unwrap();
    assert_eq!(vector_product.shape(), &[] as &[usize]);
    assert_eq!(vector_product.data(), &[0.0]);

    let matrix_base =
        NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let empty_matrix = matrix_base.view().slice([2, 3], [0, 0]).unwrap();

    let matrix_product = matmul(empty_matrix.clone(), empty_matrix.transpose()).unwrap();
    assert_eq!(matrix_product.shape(), &[0, 0]);
    assert!(matrix_product.data().is_empty());
}

#[test]
fn shape_overflow_errors_remain_stable_across_crate_boundaries() {
    let element_count_overflow =
        AtlasNdError::ShapeOverflow { op: "element count", shape: vec![usize::MAX, 2] };
    let stride_overflow =
        AtlasNdError::ShapeOverflow { op: "stride computation", shape: vec![2, usize::MAX, 2] };

    assert_eq!(checked_element_count(&[usize::MAX, 2]).unwrap_err(), element_count_overflow);
    assert_eq!(checked_compute_strides(&[2, usize::MAX, 2]).unwrap_err(), stride_overflow);
    assert_eq!(
        AtlasStatsError::from(element_count_overflow.clone()),
        AtlasStatsError::NdArray(element_count_overflow.clone())
    );

    let mut rng = AtlasRng::seed_from_u64(8_192);

    assert_eq!(
        uniform([usize::MAX, 2], 0_i32, 10_i32, &mut rng).unwrap_err(),
        AtlasRandomError::NdArray(element_count_overflow.clone())
    );
    assert_eq!(
        normal([usize::MAX, 2], 0.0_f64, 1.0, &mut rng).unwrap_err(),
        AtlasRandomError::NdArray(element_count_overflow)
    );
}

#[test]
fn post_hardening_error_messages_remain_exact_across_crates() {
    let empty_matrix = NDArray::<i32>::new([0, 3], 1).unwrap();
    let empty_vector = NDArray::from_shape_vec([0], Vec::<f64>::new()).unwrap();
    let lhs = NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs = NDArray::from_shape_vec([2, 2], vec![7.0_f64, 8.0, 9.0, 10.0]).unwrap();
    let mut rng = AtlasRng::seed_from_u64(16_384);

    assert_eq!(
        checked_element_count(&[usize::MAX, 2]).unwrap_err().to_string(),
        format!("shape overflow for element count: {:?}", vec![usize::MAX, 2])
    );
    assert_eq!(empty_matrix.mean().unwrap_err().to_string(), "empty input for mean");
    assert_eq!(empty_matrix.min_axis(0).unwrap_err().to_string(), "empty input for min");
    assert_eq!(variance(&empty_vector).unwrap_err().to_string(), "empty input for variance");
    assert_eq!(stddev(&empty_vector).unwrap_err().to_string(), "empty input for stddev");
    assert_eq!(
        covariance(&empty_vector, &empty_vector).unwrap_err().to_string(),
        "empty input for covariance"
    );
    assert_eq!(
        normal([2], 0.0_f64, 0.0, &mut rng).unwrap_err().to_string(),
        "invalid argument for normal: stddev must be strictly positive"
    );
    assert_eq!(
        matmul(&lhs, &rhs).unwrap_err().to_string(),
        "shape mismatch for matmul: left [2, 3], right [2, 2]: left matrix column count must match right matrix row count"
    );
}

#[test]
fn transparent_ndarray_error_wrappers_preserve_variant_and_display() {
    let overflow = checked_element_count(&[usize::MAX, 2]).unwrap_err();

    let linalg_error = AtlasLinalgError::from(overflow.clone());
    let stats_error = AtlasStatsError::from(overflow.clone());
    let random_error = AtlasRandomError::from(overflow.clone());

    assert_eq!(linalg_error, AtlasLinalgError::NdArray(overflow.clone()));
    assert_eq!(stats_error, AtlasStatsError::NdArray(overflow.clone()));
    assert_eq!(random_error, AtlasRandomError::NdArray(overflow.clone()));

    assert_eq!(linalg_error.to_string(), overflow.to_string());
    assert_eq!(stats_error.to_string(), overflow.to_string());
    assert_eq!(random_error.to_string(), overflow.to_string());

    let mut rng = AtlasRng::seed_from_u64(32_768);

    assert_eq!(
        uniform([usize::MAX, 2], 0_i32, 10_i32, &mut rng).unwrap_err(),
        AtlasRandomError::NdArray(overflow.clone())
    );
    assert_eq!(
        normal([usize::MAX, 2], 0.0_f64, 1.0, &mut rng).unwrap_err(),
        AtlasRandomError::NdArray(overflow.clone())
    );
    assert_eq!(
        uniform([usize::MAX, 2], 0_i32, 10_i32, &mut rng).unwrap_err().to_string(),
        overflow.to_string()
    );
}
