use atlas_ndarray::array::NDArray;
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::time::Duration;

const BENCH_SIZES: [usize; 4] = [1 << 10, 1 << 14, 1 << 18, 1 << 20];
const LARGE_INPUT_THRESHOLD: usize = 1 << 20;
const DEFAULT_SAMPLE_SIZE: usize = 100;
const LARGE_INPUT_SAMPLE_SIZE: usize = 50;
const DEFAULT_MEASUREMENT_SECS: u64 = 5;
const LARGE_INPUT_MEASUREMENT_SECS: u64 = 10;

fn configure_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>, size: usize) {
    group.throughput(Throughput::Elements(size as u64));

    if size >= LARGE_INPUT_THRESHOLD {
        group.sample_size(LARGE_INPUT_SAMPLE_SIZE);
        group.measurement_time(Duration::from_secs(LARGE_INPUT_MEASUREMENT_SECS));
    } else {
        group.sample_size(DEFAULT_SAMPLE_SIZE);
        group.measurement_time(Duration::from_secs(DEFAULT_MEASUREMENT_SECS));
    }
}

fn bench_contiguous_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/contiguous_add");

    for size in BENCH_SIZES {
        configure_group(&mut group, size);
        let lhs = NDArray::from_vec(vec![size], vec![1.0_f64; size]).unwrap();
        let rhs = NDArray::from_vec(vec![size], vec![2.0_f64; size]).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| black_box(lhs.add(black_box(&rhs)).unwrap()))
        });
    }

    group.finish();
}

fn bench_scalar_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/scalar_add");

    for size in BENCH_SIZES {
        configure_group(&mut group, size);
        let array = NDArray::from_vec(vec![size], vec![1.0_f64; size]).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| black_box(array.add(black_box(2.0_f64))))
        });
    }

    group.finish();
}

fn bench_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/sum");

    for size in BENCH_SIZES {
        configure_group(&mut group, size);
        let array = NDArray::from_vec(vec![size], vec![1.0_f64; size]).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| black_box(array.sum()))
        });
    }

    group.finish();
}

fn bench_mean(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/mean");

    for size in BENCH_SIZES {
        configure_group(&mut group, size);
        let array = NDArray::from_vec(vec![size], vec![1.0_f64; size]).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| black_box(array.mean().unwrap()))
        });
    }

    group.finish();
}

criterion_group!(
    ndarray_contiguous_kernels,
    bench_contiguous_add,
    bench_scalar_add,
    bench_sum,
    bench_mean
);
criterion_main!(ndarray_contiguous_kernels);
