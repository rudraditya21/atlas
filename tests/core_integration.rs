use atlas_linalg::{cholesky, matmul, qr};
use atlas_ndarray::NDArray;
use atlas_random::{AtlasRng, normal, uniform};
use atlas_stats::{correlation, covariance, stddev, variance};

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
