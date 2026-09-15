#[path = "../support/mod.rs"]
mod common;

use atlas_benchmarks::ml_benchmark_fixtures::{
    FEATURE_COUNT, SAMPLE_COUNTS, classification_labels, features, regression_targets,
};
use atlas_ml::{
    BinaryLogisticRegression, KnnClassifier, KnnConfig, KnnSearchAlgorithm, LinearRegression,
    LogisticRegressionConfig, RidgeRegression, RidgeRegressionConfig,
};
use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

fn bench_knn_backend_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("ml/knn/backend_selection");

    for sample_count in SAMPLE_COUNTS {
        common::configure_timing(&mut group, sample_count * sample_count);
        let features = features(sample_count);
        let labels = classification_labels(sample_count);
        for (name, algorithm) in [
            ("brute_force", KnnSearchAlgorithm::BruteForce),
            ("kd_tree", KnnSearchAlgorithm::KdTree),
            ("ball_tree", KnnSearchAlgorithm::BallTree),
            ("auto", KnnSearchAlgorithm::Auto),
        ] {
            let config = KnnConfig::new(5).unwrap().with_search_algorithm(algorithm).unwrap();
            group.bench_with_input(BenchmarkId::new(name, sample_count), &sample_count, |b, _| {
                b.iter_batched(
                    || (features.clone(), labels.clone()),
                    |(features, labels)| {
                        black_box(KnnClassifier::fit(features, labels, config).unwrap())
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }

    group.finish();
}

fn bench_linear_and_ridge_fitting(c: &mut Criterion) {
    let mut group = c.benchmark_group("ml/regression/fit");
    let ridge_config = RidgeRegressionConfig::new(0.1).unwrap();

    for sample_count in SAMPLE_COUNTS {
        common::configure_timing(&mut group, sample_count * FEATURE_COUNT * FEATURE_COUNT);
        let features = features(sample_count);
        let targets = regression_targets(&features);
        group.bench_with_input(BenchmarkId::new("linear", sample_count), &sample_count, |b, _| {
            b.iter(|| black_box(LinearRegression::fit(&features, &targets).unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("ridge", sample_count), &sample_count, |b, _| {
            b.iter(|| black_box(RidgeRegression::fit(&features, &targets, ridge_config).unwrap()))
        });
    }

    group.finish();
}

fn bench_logistic_fitting(c: &mut Criterion) {
    let mut fit_group = c.benchmark_group("ml/logistic/fit");
    let config = LogisticRegressionConfig::new(0.25, 250, 1e-6).unwrap();

    for sample_count in SAMPLE_COUNTS {
        let features = features(sample_count);
        let labels = classification_labels(sample_count);

        common::configure_timing(
            &mut fit_group,
            sample_count * FEATURE_COUNT * config.max_iterations(),
        );
        fit_group.bench_with_input(
            BenchmarkId::from_parameter(sample_count),
            &sample_count,
            |b, _| {
                b.iter(|| {
                    black_box(BinaryLogisticRegression::fit(&features, &labels, config).unwrap())
                })
            },
        );
    }

    fit_group.finish();
}

fn bench_logistic_prediction(c: &mut Criterion) {
    let mut prediction_group = c.benchmark_group("ml/logistic/predict");
    let config = LogisticRegressionConfig::new(0.25, 250, 1e-6).unwrap();

    for sample_count in SAMPLE_COUNTS {
        let features = features(sample_count);
        let labels = classification_labels(sample_count);
        let model = BinaryLogisticRegression::fit(&features, &labels, config).unwrap();

        common::configure_timing(&mut prediction_group, sample_count * FEATURE_COUNT);
        prediction_group.bench_with_input(
            BenchmarkId::from_parameter(sample_count),
            &sample_count,
            |b, _| b.iter(|| black_box(model.predict(&features).unwrap())),
        );
    }

    prediction_group.finish();
}

criterion_group!(
    ml_baselines,
    bench_knn_backend_selection,
    bench_linear_and_ridge_fitting,
    bench_logistic_fitting,
    bench_logistic_prediction
);
criterion_main!(ml_baselines);
