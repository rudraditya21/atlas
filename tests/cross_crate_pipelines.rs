use atlas_arrow::{from_arrow_record_batch, to_arrow_record_batch};
use atlas_linalg::matmul;
use atlas_ml::{
    NearestCentroidClassifier, StandardScaler, classification_accuracy, stratified_train_test_split,
};
use atlas_ndarray::NDArray;
use atlas_polars::{from_polars_dataframe, to_polars_dataframe};
use atlas_stats::variance;

#[test]
fn logical_views_flow_through_stats_scaling_linalg_splitting_and_ml() {
    let source =
        NDArray::from_shape_vec([2, 4], vec![0.0_f64, 0.0, 10.0, 10.0, 0.0, 1.0, 10.0, 11.0])
            .unwrap();
    let features = source.view().transpose();
    let labels = NDArray::from_shape_vec([4], vec![0_usize, 0, 1, 1]).unwrap();

    assert!(variance(features.clone()).unwrap() > 0.0);

    let scaler = StandardScaler::fit(&features).unwrap();
    let scaled = scaler.transform(&features).unwrap();
    let gram = matmul(scaled.view().transpose(), &scaled).unwrap();
    let split = stratified_train_test_split(&scaled, &labels, 0.5, 101).unwrap();
    let classifier =
        NearestCentroidClassifier::fit(split.train_features(), split.train_targets()).unwrap();
    let predictions = classifier.predict(split.test_features()).unwrap();

    assert_eq!(gram.shape(), &[2, 2]);
    assert_eq!(classification_accuracy(split.test_targets(), &predictions).unwrap(), 1.0);
}

#[test]
fn arrow_polars_ml_pipeline_preserves_features_for_prediction() {
    let features =
        NDArray::from_shape_vec([4, 2], vec![0.0_f64, 0.0, 0.0, 1.0, 10.0, 10.0, 10.0, 11.0])
            .unwrap();
    let labels = NDArray::from_shape_vec([4], vec![0_usize, 0, 1, 1]).unwrap();

    let record_batch = to_arrow_record_batch(&features, &["x", "y"]).unwrap();
    let arrow_features = from_arrow_record_batch::<f64>(&record_batch).unwrap();
    let dataframe = to_polars_dataframe(&arrow_features, &["x", "y"]).unwrap();
    let polars_features = from_polars_dataframe::<f64>(&dataframe).unwrap();
    let (scaler, scaled) = StandardScaler::fit_transform(&polars_features).unwrap();
    let classifier = NearestCentroidClassifier::fit(&scaled, &labels).unwrap();
    let queries = NDArray::from_shape_vec([2, 2], vec![0.1_f64, 0.2, 9.9, 10.8]).unwrap();
    let predictions = classifier.predict(&scaler.transform(&queries).unwrap()).unwrap();

    assert_eq!(polars_features.data(), features.data());
    assert_eq!(predictions.data(), &[0, 1]);
}
