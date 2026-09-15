use atlas_ml::{
    BinaryLogisticRegression, KnnClassifier, KnnConfig, KnnSearchAlgorithm, LinearRegression,
    LogisticRegressionConfig, RidgeRegression, RidgeRegressionConfig,
};
use atlas_ndarray::NDArray;
use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

const SAMPLE_COUNTS: [usize; 3] = [64, 256, 1_024];
const FEATURE_COUNT: usize = 8;

fn features(sample_count: usize) -> NDArray<f64> {
    NDArray::from_shape_vec(
        [sample_count, FEATURE_COUNT],
        (0..sample_count * FEATURE_COUNT)
            .map(|index| {
                let sample = index / FEATURE_COUNT;
                let feature = index % FEATURE_COUNT;
                let class_offset = if sample < sample_count / 2 { -1.0 } else { 1.0 };
                if feature == 0 {
                    class_offset
                } else {
                    ((sample * (feature * 7 + 3)) % 31) as f64 / 31.0 - 0.5
                }
            })
            .collect(),
    )
    .unwrap()
}

fn classification_labels(sample_count: usize) -> NDArray<usize> {
    NDArray::from_shape_vec(
        [sample_count],
        (0..sample_count).map(|sample| usize::from(sample >= sample_count / 2)).collect(),
    )
    .unwrap()
}

fn regression_targets(features: &NDArray<f64>) -> NDArray<f64> {
    NDArray::from_shape_vec(
        [features.shape()[0]],
        (0..features.shape()[0])
            .map(|sample| {
                0.5 + (0..FEATURE_COUNT)
                    .map(|feature| {
                        features.data()[sample * FEATURE_COUNT + feature] * (feature + 1) as f64
                    })
                    .sum::<f64>()
            })
            .collect(),
    )
    .unwrap()
}

fn bench_knn_backend_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("ml/knn/backend_selection");

    for sample_count in SAMPLE_COUNTS {
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

fn bench_logistic_fitting_and_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("ml/logistic");
    let config = LogisticRegressionConfig::new(0.25, 250, 1e-6).unwrap();

    for sample_count in SAMPLE_COUNTS {
        let features = features(sample_count);
        let labels = classification_labels(sample_count);
        let model = BinaryLogisticRegression::fit(&features, &labels, config).unwrap();

        group.bench_with_input(BenchmarkId::new("fit", sample_count), &sample_count, |b, _| {
            b.iter(|| black_box(BinaryLogisticRegression::fit(&features, &labels, config).unwrap()))
        });
        group.bench_with_input(BenchmarkId::new("predict", sample_count), &sample_count, |b, _| {
            b.iter(|| black_box(model.predict(&features).unwrap()))
        });
    }

    group.finish();
}

criterion_group!(
    ml_baselines,
    bench_knn_backend_selection,
    bench_linear_and_ridge_fitting,
    bench_logistic_fitting_and_prediction
);
criterion_main!(ml_baselines);
