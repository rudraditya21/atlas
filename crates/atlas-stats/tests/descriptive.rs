use atlas_ndarray::{ArrayView, NDArray, SliceRange};
use atlas_stats::{
    AtlasStatsError, correlation, correlation_axis, correlation_matrix, covariance,
    covariance_axis, covariance_ddof, covariance_matrix, covariance_matrix_ddof, kurtosis, median,
    quantile, skewness, stddev, stddev_axis, stddev_axis_ddof, stddev_axis_keepdims,
    stddev_axis_keepdims_ddof, stddev_ddof, variance, variance_axis, variance_axis_ddof,
    variance_axis_keepdims, variance_axis_keepdims_ddof, variance_ddof, weighted_correlation,
    weighted_covariance, weighted_mean, weighted_variance,
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

fn assert_descriptive_layout_equivalence(view: ArrayView<'_, f64>) {
    let materialized = view.to_owned();

    assert_close(variance(view.clone()).unwrap(), variance(&materialized).unwrap());
    assert_close(stddev(view.clone()).unwrap(), stddev(&materialized).unwrap());
    assert_close(skewness(view.clone()).unwrap(), skewness(&materialized).unwrap());
    assert_close(kurtosis(view.clone()).unwrap(), kurtosis(&materialized).unwrap());
    assert_close(quantile(view.clone(), 0.25).unwrap(), quantile(&materialized, 0.25).unwrap());
    assert_close(median(view).unwrap(), median(&materialized).unwrap());
}

#[test]
fn variance_and_stddev_use_population_definition() {
    let values = NDArray::from_shape_vec([4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();

    assert_close(variance(&values).unwrap(), 1.25);
    assert_close(stddev(&values).unwrap(), 1.118_033_988_749_895);
}

#[test]
fn variance_and_stddev_support_degrees_of_freedom() {
    let values = NDArray::from_shape_vec([4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();

    assert_close(variance_ddof(&values, 0).unwrap(), 1.25);
    assert_close(variance_ddof(&values, 1).unwrap(), 5.0 / 3.0);
    assert_close(stddev_ddof(&values, 1).unwrap(), (5.0_f64 / 3.0).sqrt());
}

#[test]
fn degrees_of_freedom_rejects_insufficient_samples() {
    let values = NDArray::from_shape_vec([1], vec![5.0_f64]).unwrap();

    assert_close(variance_ddof(&values, 0).unwrap(), 0.0);
    assert_eq!(
        variance_ddof(&values, 1).unwrap_err(),
        AtlasStatsError::InvalidDegreesOfFreedom { op: "variance_ddof", ddof: 1, count: 1 }
    );
    assert_eq!(
        stddev_ddof(&values, 1).unwrap_err(),
        AtlasStatsError::InvalidDegreesOfFreedom { op: "stddev_ddof", ddof: 1, count: 1 }
    );
}

#[test]
fn variance_and_stddev_support_strided_views() {
    let matrix = NDArray::from_shape_vec([2, 3], vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let view = matrix.view().transpose();

    assert_close(variance(view.clone()).unwrap(), 35.0 / 12.0);
    assert_close(stddev(view).unwrap(), (35.0_f64 / 12.0).sqrt());
}

#[test]
fn scalar_descriptive_reductions_match_contiguous_materializations() {
    let source = NDArray::from_shape_vec([4, 4], (1..=16).map(f64::from).collect()).unwrap();

    assert_descriptive_layout_equivalence(source.view().slice([1, 1], [2, 3]).unwrap());
    assert_descriptive_layout_equivalence(source.view().transpose().slice([1, 0], [3, 2]).unwrap());
}

#[test]
fn scalar_pair_reductions_match_contiguous_materializations() {
    let lhs = NDArray::from_shape_vec([4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let rhs_source =
        NDArray::from_shape_vec([8], vec![2.0_f64, 0.0, 6.0, 0.0, 10.0, 0.0, 14.0, 0.0]).unwrap();
    let rhs = rhs_source.view().slice_ranges([SliceRange::new(Some(0), Some(8), 2)]).unwrap();
    let lhs_materialized = lhs.view().to_owned();
    let rhs_materialized = rhs.to_owned();

    assert_close(
        covariance(lhs.view(), rhs.clone()).unwrap(),
        covariance(&lhs_materialized, &rhs_materialized).unwrap(),
    );
    assert_close(
        correlation(lhs.view(), rhs).unwrap(),
        correlation(&lhs_materialized, &rhs_materialized).unwrap(),
    );
}

#[test]
fn scalar_pair_reductions_preserve_ddof_weights_zero_variance_and_views() {
    let lhs_source =
        NDArray::from_shape_vec([8], vec![1.0_f64, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]).unwrap();
    let rhs_source =
        NDArray::from_shape_vec([8], vec![2.0_f64, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0, 0.0]).unwrap();
    let lhs = lhs_source.view().slice_ranges([SliceRange::new(Some(0), Some(8), 2)]).unwrap();
    let rhs = rhs_source.view().slice_ranges([SliceRange::new(Some(0), Some(8), 2)]).unwrap();
    let weights = NDArray::from_shape_vec([4], vec![1.0_f64, 2.0, 0.0, 1.0]).unwrap();
    let lhs_materialized = lhs.to_owned();
    let rhs_materialized = rhs.to_owned();
    let constant = NDArray::from_shape_vec([4], vec![5.0_f64; 4]).unwrap();

    let sample_covariance = covariance_ddof(lhs.clone(), rhs.clone(), 1).unwrap();
    assert_close(sample_covariance, 10.0 / 3.0);
    assert_close(
        sample_covariance,
        covariance_ddof(&lhs_materialized, &rhs_materialized, 1).unwrap(),
    );
    assert_close(
        correlation(lhs.clone(), rhs.clone()).unwrap(),
        correlation(&lhs_materialized, &rhs_materialized).unwrap(),
    );
    assert_close(weighted_covariance(lhs.clone(), rhs.clone(), &weights).unwrap(), 2.375);
    assert_close(weighted_correlation(lhs.clone(), rhs.clone(), &weights).unwrap(), 1.0);
    assert_eq!(
        correlation(&constant, rhs).unwrap_err(),
        AtlasStatsError::ZeroVariance { op: "correlation" }
    );
}

#[test]
fn axiswise_variance_and_stddev_reduce_the_selected_axis() {
    let values = NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    assert_eq!(variance_axis(&values, 0).unwrap().data(), &[2.25, 2.25, 2.25]);
    assert_eq!(variance_axis(&values, 1).unwrap().data(), &[2.0 / 3.0, 2.0 / 3.0]);
    assert_eq!(stddev_axis(&values, -1).unwrap().data(), &[(2.0_f64 / 3.0).sqrt(); 2]);
}

#[test]
fn axiswise_variance_and_stddev_support_degrees_of_freedom() {
    let values = NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let view = values.view().transpose();

    assert_eq!(variance_axis_ddof(&values, 1, 0).unwrap().data(), &[2.0 / 3.0; 2]);
    assert_eq!(variance_axis_ddof(&values, 1, 1).unwrap().data(), &[1.0; 2]);
    assert_eq!(stddev_axis_ddof(&values, 1, 1).unwrap().data(), &[1.0; 2]);
    assert_eq!(variance_axis_ddof(view, 0, 1).unwrap().data(), &[1.0; 2]);
}

#[test]
fn axiswise_variance_and_stddev_keepdims_preserve_reduced_axes() {
    let values = NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let variance = variance_axis_keepdims(&values, -1).unwrap();
    let sample_variance = variance_axis_keepdims_ddof(&values, 1, 1).unwrap();
    let sample_stddev = stddev_axis_keepdims_ddof(&values, 1, 1).unwrap();

    assert_eq!(variance.shape(), &[2, 1]);
    assert_eq!(variance.data(), &[2.0 / 3.0; 2]);
    assert_eq!(sample_variance.shape(), &[2, 1]);
    assert_eq!(sample_variance.data(), &[1.0; 2]);
    assert_eq!(sample_stddev.shape(), &[2, 1]);
    assert_eq!(sample_stddev.data(), &[1.0; 2]);
}

#[test]
fn axiswise_keepdims_variance_validates_lanes_and_degrees_of_freedom() {
    let singleton_lanes = NDArray::from_shape_vec([2, 1], vec![1.0_f64, 2.0]).unwrap();
    let empty_lanes = NDArray::<f64>::zeros([2, 0]).unwrap();
    let empty_output = NDArray::<f64>::zeros([0, 2]).unwrap();

    assert_eq!(variance_axis_keepdims(&singleton_lanes, 1).unwrap().data(), &[0.0; 2]);
    assert_eq!(variance_axis_keepdims(&empty_output, 1).unwrap().shape(), &[0, 1]);
    assert_eq!(
        variance_axis_keepdims_ddof(&singleton_lanes, 1, 1).unwrap_err(),
        AtlasStatsError::InvalidDegreesOfFreedom {
            op: "variance_axis_keepdims_ddof",
            ddof: 1,
            count: 1,
        }
    );
    assert_eq!(
        stddev_axis_keepdims_ddof(&singleton_lanes, 1, 1).unwrap_err(),
        AtlasStatsError::InvalidDegreesOfFreedom {
            op: "stddev_axis_keepdims_ddof",
            ddof: 1,
            count: 1,
        }
    );
    assert_eq!(
        stddev_axis_keepdims(&empty_lanes, 1).unwrap_err(),
        AtlasStatsError::EmptyInput { op: "stddev_axis_keepdims" }
    );
}

#[test]
fn axiswise_degrees_of_freedom_validate_lanes() {
    let singleton_lanes = NDArray::from_shape_vec([2, 1], vec![1.0_f64, 2.0]).unwrap();
    let empty_lanes = NDArray::<f64>::zeros([2, 0]).unwrap();

    assert_eq!(variance_axis_ddof(&singleton_lanes, 1, 0).unwrap().data(), &[0.0; 2]);
    assert_eq!(
        variance_axis_ddof(&singleton_lanes, 1, 1).unwrap_err(),
        AtlasStatsError::InvalidDegreesOfFreedom { op: "variance_axis_ddof", ddof: 1, count: 1 }
    );
    assert_eq!(
        stddev_axis_ddof(&singleton_lanes, 1, 1).unwrap_err(),
        AtlasStatsError::InvalidDegreesOfFreedom { op: "stddev_axis_ddof", ddof: 1, count: 1 }
    );
    assert_eq!(
        variance_axis_ddof(&empty_lanes, 1, 0).unwrap_err(),
        AtlasStatsError::EmptyInput { op: "variance_axis_ddof" }
    );
}

#[test]
fn axiswise_variance_and_stddev_support_transposed_and_sliced_views() {
    let source = NDArray::from_shape_vec([3, 3], (0_i32..9).collect()).unwrap();
    let view = source.view().transpose().slice([1, 0], [2, 3]).unwrap();

    assert_eq!(variance_axis(view.clone(), 1).unwrap().data(), &[6.0, 6.0]);
    assert_eq!(stddev_axis(view, 1).unwrap().data(), &[6.0_f64.sqrt(); 2]);
}

#[test]
fn axis_statistics_preserve_views_axes_and_empty_lane_contracts() {
    let lhs = NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs = NDArray::from_shape_vec([2, 3], vec![2.0_f64, 4.0, 6.0, 8.0, 10.0, 12.0]).unwrap();
    let lhs_view = lhs.view().transpose();
    let rhs_view = rhs.view().transpose();
    let lhs_materialized = lhs_view.to_owned();
    let rhs_materialized = rhs_view.to_owned();
    let singleton = NDArray::from_shape_vec([2, 1], vec![1.0_f64, 2.0]).unwrap();
    let empty = NDArray::<f64>::zeros([2, 0]).unwrap();

    assert_eq!(
        variance_axis(lhs_view.clone(), -1).unwrap().data(),
        variance_axis(&lhs_materialized, -1).unwrap().data()
    );
    assert_eq!(variance_axis_keepdims(lhs_view.clone(), -1).unwrap().shape(), &[3, 1]);
    assert_eq!(
        covariance_axis(lhs_view.clone(), rhs_view.clone(), -1).unwrap().data(),
        covariance_axis(&lhs_materialized, &rhs_materialized, -1).unwrap().data()
    );
    for value in correlation_axis(lhs_view, rhs_view, -1).unwrap().data() {
        assert_close(*value, 1.0);
    }
    assert_eq!(variance_axis(&singleton, -1).unwrap().data(), &[0.0; 2]);
    assert_eq!(covariance_axis(&singleton, &singleton, -1).unwrap().data(), &[0.0; 2]);
    assert_eq!(
        correlation_axis(&singleton, &singleton, -1).unwrap_err(),
        AtlasStatsError::ZeroVariance { op: "correlation_axis" }
    );
    assert_eq!(
        variance_axis(&empty, -1).unwrap_err(),
        AtlasStatsError::EmptyInput { op: "variance_axis" }
    );
    assert_eq!(
        covariance_axis(&empty, &empty, -1).unwrap_err(),
        AtlasStatsError::EmptyInput { op: "covariance_axis" }
    );
    assert_eq!(
        correlation_axis(&empty, &empty, -1).unwrap_err(),
        AtlasStatsError::EmptyInput { op: "correlation_axis" }
    );
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
fn covariance_supports_degrees_of_freedom_and_views() {
    let lhs = NDArray::from_shape_vec([4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let rhs = NDArray::from_shape_vec([4], vec![2.0_f64, 4.0, 6.0, 8.0]).unwrap();
    let lhs_view = lhs.view().slice([1], [3]).unwrap();
    let rhs_view = rhs.view().slice([1], [3]).unwrap();

    assert_close(covariance_ddof(&lhs, &rhs, 0).unwrap(), 2.5);
    assert_close(covariance_ddof(&lhs, &rhs, 1).unwrap(), 10.0 / 3.0);
    assert_close(covariance_ddof(lhs_view, rhs_view, 1).unwrap(), 2.0);
}

#[test]
fn covariance_degrees_of_freedom_rejects_insufficient_samples() {
    let lhs = NDArray::from_shape_vec([1], vec![5.0_f64]).unwrap();
    let rhs = NDArray::from_shape_vec([1], vec![9.0_f64]).unwrap();

    assert_close(covariance_ddof(&lhs, &rhs, 0).unwrap(), 0.0);
    assert_eq!(
        covariance_ddof(&lhs, &rhs, 1).unwrap_err(),
        AtlasStatsError::InvalidDegreesOfFreedom { op: "covariance_ddof", ddof: 1, count: 1 }
    );
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
fn covariance_matrix_supports_degrees_of_freedom_and_transposed_views() {
    let observations =
        NDArray::from_shape_vec([3, 2], vec![1.0_f64, 2.0, 2.0, 4.0, 3.0, 6.0]).unwrap();
    let source =
        NDArray::from_shape_vec([2, 4], vec![0.0_f64, 1.0, 2.0, 3.0, 0.0, 2.0, 4.0, 6.0]).unwrap();
    let view = source.view().transpose().slice([1, 0], [3, 2]).unwrap();

    assert_eq!(covariance_matrix_ddof(&observations, 1).unwrap().data(), &[1.0, 2.0, 2.0, 4.0]);
    assert_eq!(covariance_matrix_ddof(view, 1).unwrap().data(), &[1.0, 2.0, 2.0, 4.0]);
}

#[test]
fn covariance_matrix_degrees_of_freedom_validate_observations() {
    let observations =
        NDArray::from_shape_vec([3, 2], vec![1.0_f64, 2.0, 2.0, 4.0, 3.0, 6.0]).unwrap();
    let empty = NDArray::<f64>::zeros([0, 2]).unwrap();

    assert_eq!(
        covariance_matrix_ddof(&observations, 3).unwrap_err(),
        AtlasStatsError::InvalidDegreesOfFreedom {
            op: "covariance_matrix_ddof",
            ddof: 3,
            count: 3,
        }
    );
    assert_eq!(
        covariance_matrix_ddof(&empty, 0).unwrap_err(),
        AtlasStatsError::EmptyInput { op: "covariance_matrix_ddof" }
    );
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
fn weighted_statistics_use_population_weight_normalization() {
    let values = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 5.0]).unwrap();
    let rhs = NDArray::from_shape_vec([3], vec![2.0_f64, 4.0, 10.0]).unwrap();
    let weights = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 1.0]).unwrap();

    assert_close(weighted_mean(&values, &weights).unwrap(), 2.5);
    assert_close(weighted_variance(&values, &weights).unwrap(), 2.25);
    assert_close(weighted_covariance(&values, &rhs, &weights).unwrap(), 4.5);
}

#[test]
fn weighted_statistics_support_views_and_validate_weights() {
    let values_source = NDArray::from_shape_vec([4], vec![0.0_f64, 1.0, 2.0, 5.0]).unwrap();
    let weights_source = NDArray::from_shape_vec([4], vec![0.0_f64, 1.0, 2.0, 1.0]).unwrap();
    let values = values_source.view().slice([1], [3]).unwrap();
    let weights = weights_source.view().slice([1], [3]).unwrap();
    let invalid = NDArray::from_shape_vec([3], vec![1.0_f64, -1.0, 1.0]).unwrap();
    let zero = NDArray::<f64>::zeros([3]).unwrap();

    assert_close(weighted_mean(values, weights).unwrap(), 2.5);
    assert_eq!(
        weighted_mean(&invalid, &invalid).unwrap_err(),
        AtlasStatsError::InvalidWeights {
            op: "weighted_mean",
            reason: "weights must be finite and non-negative",
        }
    );
    assert_eq!(
        weighted_variance(&invalid, &zero).unwrap_err(),
        AtlasStatsError::InvalidWeights {
            op: "weighted_variance",
            reason: "weights must have a positive finite sum",
        }
    );
}

#[test]
fn quantile_uses_linear_interpolation_and_median_reuses_it() {
    let values = NDArray::from_shape_vec([4], vec![4.0_f64, 1.0, 3.0, 2.0]).unwrap();

    assert_close(quantile(&values, 0.25).unwrap(), 1.75);
    assert_close(quantile(&values, 0.5).unwrap(), 2.5);
    assert_close(median(&values).unwrap(), 2.5);
}

#[test]
fn quantile_supports_views_and_has_explicit_nan_and_validation_behavior() {
    let source = NDArray::from_shape_vec([4], vec![0.0_f64, 1.0, 3.0, 5.0]).unwrap();
    let view = source.view().slice([1], [3]).unwrap();
    let nan = NDArray::from_shape_vec([2], vec![1.0_f64, f64::NAN]).unwrap();
    let empty = NDArray::<f64>::zeros([0]).unwrap();

    assert_close(quantile(view, 0.5).unwrap(), 3.0);
    assert!(median(&nan).unwrap().is_nan());
    assert_eq!(quantile(&empty, 0.5).unwrap_err(), AtlasStatsError::EmptyInput { op: "quantile" });
    assert_eq!(median(&empty).unwrap_err(), AtlasStatsError::EmptyInput { op: "median" });
    assert_eq!(
        quantile(&source, 1.1).unwrap_err(),
        AtlasStatsError::InvalidQuantile { reason: "must be finite and within [0, 1]" }
    );
}

#[test]
fn axiswise_quantile_and_median_support_keepdims_and_views() {
    use atlas_stats::{median_axis, median_axis_keepdims, quantile_axis, quantile_axis_keepdims};

    let values = NDArray::from_shape_vec([2, 3], vec![1.0_f64, 3.0, 5.0, 2.0, 4.0, 6.0]).unwrap();
    let view = values.view().transpose();

    assert_eq!(quantile_axis(&values, 0.5, 1).unwrap().data(), &[3.0, 4.0]);
    assert_eq!(median_axis(view, 1).unwrap().data(), &[1.5, 3.5, 5.5]);
    assert_eq!(quantile_axis_keepdims(&values, 0.5, 1).unwrap().shape(), &[2, 1]);
    assert_eq!(median_axis_keepdims(&values, 0).unwrap().shape(), &[1, 3]);
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
