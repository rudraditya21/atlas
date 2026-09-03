use atlas_ndarray::NDArray;
use atlas_stats::{
    AtlasStatsError, correlation_matrix, covariance, covariance_matrix, median, median_axis,
    quantile, quantile_axis, stddev, stddev_axis, variance, variance_axis, weighted_mean,
};

fn assert_close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() <= 1.0e-12, "expected {expected}, got {actual}");
}

#[test]
fn descriptive_statistics_match_integer_and_float_references() {
    let integers = NDArray::from_shape_vec([4], vec![1_i32, 2, 3, 4]).unwrap();
    let floats = NDArray::from_shape_vec([4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let weights = NDArray::from_shape_vec([4], vec![1.0_f64, 2.0, 1.0, 2.0]).unwrap();

    assert_close(variance(&integers).unwrap(), 1.25);
    assert_close(stddev(&floats).unwrap(), 1.118_033_988_749_895);
    assert_close(quantile(&integers, 0.25).unwrap(), 1.75);
    assert_close(median(&floats).unwrap(), 2.5);
    assert_close(weighted_mean(&floats, &weights).unwrap(), 8.0 / 3.0);
}

#[test]
fn descriptive_statistics_propagate_nan_values() {
    let values = NDArray::from_shape_vec([3], vec![1.0_f64, f64::NAN, 3.0]).unwrap();
    let paired = NDArray::from_shape_vec([3], vec![2.0_f64, 4.0, 6.0]).unwrap();
    let matrix =
        NDArray::from_shape_vec([3, 2], vec![1.0_f64, 2.0, f64::NAN, 4.0, 3.0, 6.0]).unwrap();

    assert!(variance(&values).unwrap().is_nan());
    assert!(covariance(&values, &paired).unwrap().is_nan());
    assert!(median(&values).unwrap().is_nan());
    assert!(covariance_matrix(&matrix).unwrap().data().iter().any(|value| value.is_nan()));
    assert!(correlation_matrix(&matrix).unwrap().data().iter().any(|value| value.is_nan()));
}

#[test]
fn descriptive_statistics_report_empty_inputs() {
    let empty = NDArray::<f64>::zeros([0]).unwrap();
    let empty_axis = NDArray::<f64>::zeros([2, 0]).unwrap();

    assert_eq!(variance(&empty).unwrap_err(), AtlasStatsError::EmptyInput { op: "variance" });
    assert_eq!(median(&empty).unwrap_err(), AtlasStatsError::EmptyInput { op: "median" });
    assert_eq!(
        quantile_axis(&empty_axis, 0.5, 1).unwrap_err(),
        AtlasStatsError::EmptyInput { op: "quantile_axis" }
    );
}

#[test]
fn descriptive_statistics_match_for_views_and_axis_reductions() {
    let source = NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let view = source.view().transpose();

    assert_eq!(variance_axis(&source, 1).unwrap().data(), &[2.0 / 3.0; 2]);
    assert_eq!(stddev_axis(view.clone(), 1).unwrap().data(), &[1.5, 1.5, 1.5]);
    assert_eq!(median_axis(view, 1).unwrap().data(), &[2.5, 3.5, 4.5]);
}
