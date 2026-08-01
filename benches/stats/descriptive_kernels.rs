#[path = "../support/mod.rs"]
mod common;

use atlas_ndarray::NDArray;
use atlas_stats::{correlation, covariance, stddev, variance};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

fn filled_vector(size: usize, scale: f64) -> NDArray<f64> {
    let mut data = Vec::with_capacity(size);

    for index in 0..size {
        data.push((index as f64 + 1.0) * scale);
    }

    NDArray::from_shape_vec([size], data).unwrap()
}

fn filled_square_matrix(side: usize, scale: f64) -> NDArray<f64> {
    let mut data = Vec::with_capacity(side * side);

    for index in 0..(side * side) {
        data.push((index as f64 + 1.0) * scale);
    }

    NDArray::from_shape_vec([side, side], data).unwrap()
}

fn padded_square_source(side: usize, scale: f64) -> NDArray<f64> {
    let padded_cols = side + 1;
    let mut data = Vec::with_capacity(side * padded_cols);

    for index in 0..(side * padded_cols) {
        data.push((index as f64 + 1.0) * scale);
    }

    NDArray::from_shape_vec([side, padded_cols], data).unwrap()
}

fn bench_variance(c: &mut Criterion) {
    let mut group = c.benchmark_group("stats/variance/contiguous");

    for size in common::VECTOR_SIZES {
        let values = filled_vector(size, 0.5);

        common::configure_group(&mut group, size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| variance(black_box(&values)).unwrap())
        });
    }

    group.finish();
}

fn bench_stddev(c: &mut Criterion) {
    let mut group = c.benchmark_group("stats/stddev/contiguous");

    for size in common::VECTOR_SIZES {
        let values = filled_vector(size, 0.5);

        common::configure_group(&mut group, size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| stddev(black_box(&values)).unwrap())
        });
    }

    group.finish();
}

fn bench_covariance(c: &mut Criterion) {
    let mut group = c.benchmark_group("stats/covariance/contiguous");

    for size in common::VECTOR_SIZES {
        let lhs = filled_vector(size, 0.5);
        let rhs = filled_vector(size, 1.0);

        common::configure_group(&mut group, size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| covariance(black_box(&lhs), black_box(&rhs)).unwrap())
        });
    }

    group.finish();
}

fn bench_correlation(c: &mut Criterion) {
    let mut group = c.benchmark_group("stats/correlation/contiguous");

    for size in common::VECTOR_SIZES {
        let lhs = filled_vector(size, 0.5);
        let rhs = filled_vector(size, 1.0);

        common::configure_group(&mut group, size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| correlation(black_box(&lhs), black_box(&rhs)).unwrap())
        });
    }

    group.finish();
}

fn bench_variance_strided(c: &mut Criterion) {
    let mut group = c.benchmark_group("stats/variance/strided");

    for side in common::SQUARE_MATRIX_SIDES {
        let elements = side * side;
        let label = common::square_label(side);
        let contiguous = filled_square_matrix(side, 0.5);
        let transposed = contiguous.view().transpose();
        let padded = padded_square_source(side, 0.5);
        let sliced = padded.view().slice([0, 0], [side, side]).unwrap();

        common::configure_group(&mut group, elements);
        group.bench_with_input(BenchmarkId::new("transpose", &label), &elements, |b, _| {
            b.iter(|| variance(black_box(transposed.clone())).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("slice", &label), &elements, |b, _| {
            b.iter(|| variance(black_box(sliced.clone())).unwrap())
        });
    }

    group.finish();
}

fn bench_stddev_strided(c: &mut Criterion) {
    let mut group = c.benchmark_group("stats/stddev/strided");

    for side in common::SQUARE_MATRIX_SIDES {
        let elements = side * side;
        let label = common::square_label(side);
        let contiguous = filled_square_matrix(side, 0.5);
        let transposed = contiguous.view().transpose();
        let padded = padded_square_source(side, 0.5);
        let sliced = padded.view().slice([0, 0], [side, side]).unwrap();

        common::configure_group(&mut group, elements);
        group.bench_with_input(BenchmarkId::new("transpose", &label), &elements, |b, _| {
            b.iter(|| stddev(black_box(transposed.clone())).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("slice", &label), &elements, |b, _| {
            b.iter(|| stddev(black_box(sliced.clone())).unwrap())
        });
    }

    group.finish();
}

criterion_group!(
    stats_descriptive_kernels,
    bench_variance,
    bench_variance_strided,
    bench_stddev,
    bench_stddev_strided,
    bench_covariance,
    bench_correlation
);
criterion_main!(stats_descriptive_kernels);
