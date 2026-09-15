use atlas_benchmarks::ml_benchmark_fixtures::{
    SAMPLE_COUNTS, classification_labels, features, regression_targets,
};
use atlas_ml::{
    AUTO_BRUTE_FORCE_MAX_SAMPLES, BinaryLogisticRegression, KnnClassifier, KnnConfig,
    KnnSearchAlgorithm, LinearRegression, LogisticRegressionConfig, RidgeRegression,
    RidgeRegressionConfig,
};

fn assert_close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() <= 1e-9, "expected {expected}, got {actual}");
}

#[test]
fn knn_backend_benchmark_fixtures_are_equivalent() {
    for sample_count in SAMPLE_COUNTS {
        let features = features(sample_count);
        let labels = classification_labels(sample_count);
        let mut predictions = None;

        for algorithm in [
            KnnSearchAlgorithm::BruteForce,
            KnnSearchAlgorithm::KdTree,
            KnnSearchAlgorithm::BallTree,
            KnnSearchAlgorithm::Auto,
        ] {
            let config = KnnConfig::new(5).unwrap().with_search_algorithm(algorithm).unwrap();
            let model = KnnClassifier::fit(features.clone(), labels.clone(), config).unwrap();
            let actual = model.predict(&features).unwrap();
            let expected_backend = match algorithm {
                KnnSearchAlgorithm::Auto if sample_count <= AUTO_BRUTE_FORCE_MAX_SAMPLES => {
                    KnnSearchAlgorithm::BruteForce
                }
                KnnSearchAlgorithm::Auto => KnnSearchAlgorithm::KdTree,
                algorithm => algorithm,
            };

            assert_eq!(model.selected_search_algorithm(), expected_backend);
            if let Some(expected) = predictions.as_ref() {
                assert_eq!(actual.data(), expected);
            } else {
                predictions = Some(actual.data().to_vec());
            }
        }
    }
}

#[test]
fn linear_and_ridge_benchmark_fixtures_produce_valid_predictions() {
    let ridge_config = RidgeRegressionConfig::new(0.1).unwrap();

    for sample_count in SAMPLE_COUNTS {
        let features = features(sample_count);
        let targets = regression_targets(&features);
        let linear = LinearRegression::fit(&features, &targets).unwrap();
        let ridge = RidgeRegression::fit(&features, &targets, ridge_config).unwrap();
        let linear_predictions = linear.predict(&features).unwrap();
        let ridge_predictions = ridge.predict(&features).unwrap();

        for (actual, expected) in linear_predictions.data().iter().zip(targets.data()) {
            assert_close(*actual, *expected);
        }
        assert_eq!(ridge_predictions.shape(), targets.shape());
        assert!(ridge_predictions.data().iter().all(|value| value.is_finite()));
    }
}

#[test]
fn logistic_benchmark_fixtures_predict_their_training_labels() {
    let config = LogisticRegressionConfig::new(0.25, 250, 1e-6).unwrap();

    for sample_count in SAMPLE_COUNTS {
        let features = features(sample_count);
        let labels = classification_labels(sample_count);
        let model = BinaryLogisticRegression::fit(&features, &labels, config).unwrap();

        assert_eq!(model.predict(&features).unwrap().data(), labels.data());
    }
}
