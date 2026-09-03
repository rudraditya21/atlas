use atlas_ndarray::NDArray;
use atlas_stats::{
    AtlasStatsError, correlation, correlation_matrix, covariance, covariance_matrix, stddev,
    stddev_axis, variance, variance_axis,
};

fn assert_close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() <= 1e-10);
}

fn assert_result_close(
    actual: Result<f64, AtlasStatsError>,
    expected: Result<f64, AtlasStatsError>,
) {
    match (actual, expected) {
        (Ok(actual), Ok(expected)) => assert_close(actual, expected),
        (Err(actual), Err(expected)) => assert_eq!(actual, expected),
        (actual, expected) => panic!("result mismatch: actual={actual:?}, expected={expected:?}"),
    }
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
fn axiswise_variance_and_stddev_reduce_the_selected_axis() {
    let values = NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    assert_eq!(variance_axis(&values, 0).unwrap().data(), &[2.25, 2.25, 2.25]);
    assert_eq!(variance_axis(&values, 1).unwrap().data(), &[2.0 / 3.0, 2.0 / 3.0]);
    assert_eq!(stddev_axis(&values, -1).unwrap().data(), &[(2.0_f64 / 3.0).sqrt(); 2]);
}

#[test]
fn axiswise_variance_and_stddev_support_transposed_and_sliced_views() {
    let source = NDArray::from_shape_vec([3, 3], (0_i32..9).collect()).unwrap();
    let view = source.view().transpose().slice([1, 0], [2, 3]).unwrap();

    assert_eq!(variance_axis(view.clone(), 1).unwrap().data(), &[6.0, 6.0]);
    assert_eq!(stddev_axis(view, 1).unwrap().data(), &[6.0_f64.sqrt(); 2]);
}

#[test]
fn axiswise_variance_validates_axes_and_empty_reduction_lanes() {
    let scalar = NDArray::from_shape_vec([], vec![1.0_f64]).unwrap();
    let empty_lane = NDArray::<f64>::zeros([2, 0]).unwrap();
    let empty_output = NDArray::<f64>::zeros([0, 2]).unwrap();

    assert_eq!(
        variance_axis(&scalar, 0).unwrap_err(),
        AtlasStatsError::NdArray(atlas_ndarray::AtlasNdError::InvalidAxis { axis: 0, ndim: 0 })
    );
    assert_eq!(
        variance_axis(&empty_lane, 1).unwrap_err(),
        AtlasStatsError::EmptyInput { op: "variance_axis" }
    );
    assert_eq!(
        stddev_axis(&empty_lane, 1).unwrap_err(),
        AtlasStatsError::EmptyInput { op: "stddev_axis" }
    );
    assert_eq!(variance_axis(&empty_output, 1).unwrap().shape(), &[0]);
}

#[test]
fn covariance_uses_population_definition() {
    let lhs = NDArray::from_shape_vec([4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let rhs = NDArray::from_shape_vec([4], vec![2.0_f64, 4.0, 6.0, 8.0]).unwrap();

    assert_close(covariance(&lhs, &rhs).unwrap(), 2.5);
}

#[test]
fn covariance_matrix_uses_observations_as_rows_and_variables_as_columns() {
    let observations =
        NDArray::from_shape_vec([3, 2], vec![1.0_f64, 2.0, 2.0, 4.0, 3.0, 6.0]).unwrap();

    let covariance = covariance_matrix(&observations).unwrap();

    assert_eq!(covariance.shape(), &[2, 2]);
    assert_eq!(covariance.data(), &[2.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0, 8.0 / 3.0]);
}

#[test]
fn covariance_matrix_supports_transposed_and_sliced_views() {
    let source =
        NDArray::from_shape_vec([2, 4], vec![0.0_f64, 1.0, 2.0, 3.0, 0.0, 2.0, 4.0, 6.0]).unwrap();
    let view = source.view().transpose().slice([1, 0], [3, 2]).unwrap();

    assert_eq!(
        covariance_matrix(view).unwrap().data(),
        &[2.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0, 8.0 / 3.0]
    );
}

#[test]
fn covariance_matrix_validates_rank_and_empty_observations() {
    let vector = NDArray::from_shape_vec([2], vec![1.0_f64, 2.0]).unwrap();
    let empty = NDArray::<f64>::zeros([0, 2]).unwrap();
    let no_variables = NDArray::<f64>::zeros([2, 0]).unwrap();

    assert_eq!(
        covariance_matrix(&vector).unwrap_err(),
        AtlasStatsError::InvalidInputRank {
            op: "covariance_matrix",
            expected: "rank-2 [observations, variables] matrix",
            rank: 1,
        }
    );
    assert_eq!(
        covariance_matrix(&empty).unwrap_err(),
        AtlasStatsError::EmptyInput { op: "covariance_matrix" }
    );
    assert_eq!(covariance_matrix(&no_variables).unwrap().shape(), &[0, 0]);
}

#[test]
fn correlation_matrix_returns_pairwise_pearson_correlations() {
    let observations =
        NDArray::from_shape_vec([4, 2], vec![1.0_f64, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0]).unwrap();

    assert_eq!(correlation_matrix(&observations).unwrap().data(), &[1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn correlation_matrix_supports_views_and_rejects_zero_variance_variables() {
    let source =
        NDArray::from_shape_vec([2, 4], vec![0.0_f64, 1.0, 2.0, 3.0, 0.0, 2.0, 4.0, 6.0]).unwrap();
    let view = source.view().transpose().slice([1, 0], [3, 2]).unwrap();
    let constant = NDArray::from_shape_vec([3, 2], vec![1.0_f64, 1.0, 1.0, 2.0, 1.0, 3.0]).unwrap();

    assert_eq!(correlation_matrix(view).unwrap().data(), &[1.0, 1.0, 1.0, 1.0]);
    assert_eq!(
        correlation_matrix(&constant).unwrap_err(),
        AtlasStatsError::ZeroVariance { op: "correlation_matrix" }
    );
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
fn variance_and_stddev_match_between_owned_arrays_and_strided_views() {
    let owned = NDArray::from_shape_vec([3, 2], vec![0.0_f64, 3.0, 1.0, 4.0, 2.0, 5.0]).unwrap();
    let base = NDArray::from_shape_vec([2, 3], vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let view = base.view().transpose();

    assert_result_close(variance(&owned), variance(view.clone()));
    assert_result_close(stddev(&owned), stddev(view));
}

#[test]
fn covariance_and_correlation_match_between_owned_arrays_and_views() {
    let lhs_owned = NDArray::from_shape_vec([4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let rhs_owned = NDArray::from_shape_vec([4], vec![2.0_f64, 4.0, 6.0, 8.0]).unwrap();
    let lhs_base = NDArray::from_shape_vec([5], vec![0.0_f64, 1.0, 2.0, 3.0, 4.0]).unwrap();
    let rhs_base = NDArray::from_shape_vec([5], vec![0.0_f64, 2.0, 4.0, 6.0, 8.0]).unwrap();
    let lhs_view = lhs_base.view().slice([1], [4]).unwrap();
    let rhs_view = rhs_base.view().slice([1], [4]).unwrap();

    assert_result_close(
        covariance(&lhs_owned, &rhs_owned),
        covariance(lhs_view.clone(), rhs_view.clone()),
    );
    assert_result_close(correlation(&lhs_owned, &rhs_owned), correlation(lhs_view, rhs_view));
}

#[test]
fn stats_validation_errors_match_between_owned_arrays_and_views() {
    let matrix_owned = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let matrix_view = matrix_owned.view().transpose();
    let vector_owned = NDArray::from_shape_vec([2], vec![5.0_f64, 6.0]).unwrap();
    let lhs_owned = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let rhs_owned = NDArray::from_shape_vec([2], vec![4.0_f64, 5.0]).unwrap();
    let lhs_view_base = NDArray::from_shape_vec([4], vec![0.0_f64, 1.0, 2.0, 3.0]).unwrap();
    let rhs_view_base = NDArray::from_shape_vec([3], vec![0.0_f64, 4.0, 5.0]).unwrap();
    let lhs_view = lhs_view_base.view().slice([1], [3]).unwrap();
    let rhs_view = rhs_view_base.view().slice([1], [2]).unwrap();
    let constant_owned = NDArray::from_shape_vec([3], vec![7.0_f64, 7.0, 7.0]).unwrap();
    let constant_view_base = NDArray::from_shape_vec([4], vec![0.0_f64, 7.0, 7.0, 7.0]).unwrap();
    let constant_view = constant_view_base.view().slice([1], [3]).unwrap();

    assert_eq!(
        covariance(&matrix_owned, &vector_owned).unwrap_err(),
        covariance(matrix_view, &vector_owned).unwrap_err()
    );
    assert_eq!(
        correlation(&lhs_owned, &rhs_owned).unwrap_err(),
        correlation(lhs_view, rhs_view).unwrap_err()
    );
    assert_eq!(
        correlation(&constant_owned, &lhs_owned).unwrap_err(),
        correlation(constant_view, lhs_view_base.view().slice([1], [3]).unwrap()).unwrap_err()
    );
}

#[test]
fn singleton_population_statistics_reduce_to_zero_except_correlation() {
    let lhs = NDArray::from_shape_vec([1], vec![5.0_f64]).unwrap();
    let rhs = NDArray::from_shape_vec([1], vec![9.0_f64]).unwrap();

    assert_close(variance(&lhs).unwrap(), 0.0);
    assert_close(stddev(&lhs).unwrap(), 0.0);
    assert_close(covariance(&lhs, &rhs).unwrap(), 0.0);
    assert_eq!(
        correlation(&lhs, &rhs).unwrap_err(),
        AtlasStatsError::ZeroVariance { op: "correlation" }
    );
}

#[test]
fn descriptive_stats_report_exact_empty_input_errors() {
    let empty = NDArray::from_shape_vec([0], Vec::<f64>::new()).unwrap();

    assert_eq!(variance(&empty).unwrap_err(), AtlasStatsError::EmptyInput { op: "variance" });
    assert_eq!(stddev(&empty).unwrap_err(), AtlasStatsError::EmptyInput { op: "stddev" });
}

#[test]
fn descriptive_stats_report_exact_rank_mismatch_errors() {
    let matrix = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let vector = NDArray::from_shape_vec([2], vec![5.0_f64, 6.0]).unwrap();

    assert_eq!(
        covariance(&matrix, &vector).unwrap_err(),
        AtlasStatsError::InvalidInputRank { op: "covariance", expected: "rank-1 vector", rank: 2 }
    );
    assert_eq!(
        covariance(&vector, &matrix).unwrap_err(),
        AtlasStatsError::InvalidInputRank { op: "covariance", expected: "rank-1 vector", rank: 2 }
    );
    assert_eq!(
        correlation(&matrix, &vector).unwrap_err(),
        AtlasStatsError::InvalidInputRank { op: "correlation", expected: "rank-1 vector", rank: 2 }
    );
    assert_eq!(
        correlation(&vector, &matrix).unwrap_err(),
        AtlasStatsError::InvalidInputRank { op: "correlation", expected: "rank-1 vector", rank: 2 }
    );
}

#[test]
fn descriptive_stats_report_exact_length_mismatch_errors() {
    let lhs = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let rhs = NDArray::from_shape_vec([2], vec![4.0_f64, 5.0]).unwrap();

    assert_eq!(
        covariance(&lhs, &rhs).unwrap_err(),
        AtlasStatsError::ShapeMismatch {
            op: "covariance",
            left: vec![3],
            right: vec![2],
            reason: "vector lengths must match",
        }
    );
    assert_eq!(
        correlation(&lhs, &rhs).unwrap_err(),
        AtlasStatsError::ShapeMismatch {
            op: "correlation",
            left: vec![3],
            right: vec![2],
            reason: "vector lengths must match",
        }
    );
}

#[test]
fn descriptive_stats_report_exact_zero_variance_correlation_errors() {
    let constant = NDArray::from_shape_vec([3], vec![7.0_f64, 7.0, 7.0]).unwrap();
    let lhs = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();

    assert_eq!(
        correlation(&constant, &lhs).unwrap_err(),
        AtlasStatsError::ZeroVariance { op: "correlation" }
    );
    assert_eq!(
        correlation(&lhs, &constant).unwrap_err(),
        AtlasStatsError::ZeroVariance { op: "correlation" }
    );
}
