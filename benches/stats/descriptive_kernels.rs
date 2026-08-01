use atlas_ndarray::NDArray;
use atlas_stats::{correlation, covariance, stddev, variance};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

const VECTOR_SIZES: [usize; 4] = [1 << 10, 1 << 14, 1 << 18, 1 << 20];

fn filled_vector(size: usize, scale: f64) -> NDArray<f64> {
    let mut data = Vec::with_capacity(size);

    for index in 0..size {
        data.push((index as f64 + 1.0) * scale);
    }

    NDArray::from_shape_vec([size], data).unwrap()
}

fn bench_variance(c: &mut Criterion) {
    let mut group = c.benchmark_group("stats/variance");

    for size in VECTOR_SIZES {
        let values = filled_vector(size, 0.5);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| variance(black_box(&values)).unwrap())
        });
    }

    group.finish();
}

fn bench_stddev(c: &mut Criterion) {
    let mut group = c.benchmark_group("stats/stddev");

    for size in VECTOR_SIZES {
        let values = filled_vector(size, 0.5);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| stddev(black_box(&values)).unwrap())
        });
    }

    group.finish();
}

fn bench_covariance(c: &mut Criterion) {
    let mut group = c.benchmark_group("stats/covariance");

    for size in VECTOR_SIZES {
        let lhs = filled_vector(size, 0.5);
        let rhs = filled_vector(size, 1.0);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| covariance(black_box(&lhs), black_box(&rhs)).unwrap())
        });
    }

    group.finish();
}

fn bench_correlation(c: &mut Criterion) {
    let mut group = c.benchmark_group("stats/correlation");

    for size in VECTOR_SIZES {
        let lhs = filled_vector(size, 0.5);
        let rhs = filled_vector(size, 1.0);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| correlation(black_box(&lhs), black_box(&rhs)).unwrap())
        });
    }

    group.finish();
}

criterion_group!(
    stats_descriptive_kernels,
    bench_variance,
    bench_stddev,
    bench_covariance,
    bench_correlation
);
criterion_main!(stats_descriptive_kernels);
