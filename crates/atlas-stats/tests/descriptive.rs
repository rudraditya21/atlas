use atlas_ndarray::NDArray;
use atlas_stats::{AtlasStatsError, correlation, covariance, stddev, variance};

fn assert_close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() <= 1e-10);
}

#[test]
fn variance_and_stddev_use_population_definition() {
    let values = NDArray::from_shape_vec([4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();

    assert_close(variance(&values).unwrap(), 1.25);
    assert_close(stddev(&values).unwrap(), 1.118_033_988_749_895);
}

#[test]
fn variance_and_stddev_support_strided_views() {
    let matrix = NDArray::from_shape_vec([2, 3], vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let view = matrix.view().transpose();

    assert_close(variance(view.clone()).unwrap(), 35.0 / 12.0);
    assert_close(stddev(view).unwrap(), (35.0_f64 / 12.0).sqrt());
}

#[test]
fn covariance_uses_population_definition() {
    let lhs = NDArray::from_shape_vec([4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let rhs = NDArray::from_shape_vec([4], vec![2.0_f64, 4.0, 6.0, 8.0]).unwrap();

    assert_close(covariance(&lhs, &rhs).unwrap(), 2.5);
}

#[test]
fn correlation_uses_pearson_centered_definition() {
    let lhs = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let rhs = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 4.0]).unwrap();

    assert_close(correlation(&lhs, &rhs).unwrap(), 0.981_980_506_061_965_7);
}

#[test]
fn covariance_and_correlation_support_vector_views() {
    let lhs_base = NDArray::from_shape_vec([5], vec![0.0_f64, 1.0, 2.0, 3.0, 4.0]).unwrap();
    let rhs_base = NDArray::from_shape_vec([5], vec![0.0_f64, 2.0, 4.0, 6.0, 8.0]).unwrap();

    let lhs = lhs_base.view().slice([1], [4]).unwrap();
    let rhs = rhs_base.view().slice([1], [4]).unwrap();

    assert_close(covariance(lhs.clone(), rhs.clone()).unwrap(), 2.5);
    assert_close(correlation(lhs, rhs).unwrap(), 1.0);
}

#[test]
fn singleton_population_statistics_reduce_to_zero_except_correlation() {
    let lhs = NDArray::from_shape_vec([1], vec![5.0_f64]).unwrap();
    let rhs = NDArray::from_shape_vec([1], vec![9.0_f64]).unwrap();

    assert_close(variance(&lhs).unwrap(), 0.0);
    assert_close(stddev(&lhs).unwrap(), 0.0);
    assert_close(covariance(&lhs, &rhs).unwrap(), 0.0);
    assert!(matches!(
        correlation(&lhs, &rhs).unwrap_err(),
        AtlasStatsError::ZeroVariance { op: "correlation" }
    ));
}

#[test]
fn descriptive_stats_report_errors_cleanly() {
    let matrix = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let lhs = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let rhs = NDArray::from_shape_vec([2], vec![4.0_f64, 5.0]).unwrap();

    assert!(matches!(
        covariance(&matrix, &matrix).unwrap_err(),
        AtlasStatsError::InvalidInputRank { op: "covariance", .. }
    ));
    assert!(matches!(
        correlation(&lhs, &rhs).unwrap_err(),
        AtlasStatsError::ShapeMismatch { op: "correlation", .. }
    ));
}
