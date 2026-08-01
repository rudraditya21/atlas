#[path = "../support/mod.rs"]
mod common;

use atlas_linalg::{dot, matmul};
use atlas_ndarray::NDArray;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

fn padded_square_source(side: usize, value: f64) -> NDArray<f64> {
    NDArray::from_shape_vec([side, side + 1], vec![value; side * (side + 1)]).unwrap()
}

fn bench_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/dot/contiguous");

    for &size in &common::VECTOR_SIZES {
        let lhs = NDArray::from_shape_vec([size], vec![1.0_f64; size]).unwrap();
        let rhs = NDArray::from_shape_vec([size], vec![2.0_f64; size]).unwrap();

        common::configure_group(&mut group, size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| dot(black_box(&lhs), black_box(&rhs)).unwrap())
        });
    }

    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/matmul/contiguous");

    for &size in &common::DENSE_MATMUL_SIDES {
        let lhs = NDArray::from_shape_vec([size, size], vec![1.0_f64; size * size]).unwrap();
        let rhs = NDArray::from_shape_vec([size, size], vec![2.0_f64; size * size]).unwrap();

        let work_items = size * size * size;
        common::configure_group(&mut group, work_items);
        group.bench_with_input(
            BenchmarkId::from_parameter(common::square_label(size)),
            &size,
            |b, _| b.iter(|| matmul(black_box(&lhs), black_box(&rhs)).unwrap()),
        );
    }

    group.finish();
}

fn bench_matmul_rhs_transposed(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/matmul/strided_transpose");

    for &size in &common::DENSE_MATMUL_SIDES {
        let lhs = NDArray::from_shape_vec([size, size], vec![1.0_f64; size * size]).unwrap();
        let rhs_base = NDArray::from_shape_vec([size, size], vec![2.0_f64; size * size]).unwrap();

        let work_items = size * size * size;
        common::configure_group(&mut group, work_items);
        group.bench_with_input(
            BenchmarkId::from_parameter(common::square_label(size)),
            &size,
            |b, _| {
                b.iter(|| matmul(black_box(&lhs), black_box(rhs_base.view().transpose())).unwrap())
            },
        );
    }

    group.finish();
}

fn bench_matmul_rhs_strided_slice(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/matmul/strided_slice");

    for &size in &common::DENSE_MATMUL_SIDES {
        let lhs = NDArray::from_shape_vec([size, size], vec![1.0_f64; size * size]).unwrap();
        let rhs_base = padded_square_source(size, 2.0_f64);
        let rhs = rhs_base.view().slice([0, 0], [size, size]).unwrap();

        let work_items = size * size * size;
        common::configure_group(&mut group, work_items);
        group.bench_with_input(
            BenchmarkId::from_parameter(common::square_label(size)),
            &size,
            |b, _| b.iter(|| matmul(black_box(&lhs), black_box(rhs.clone())).unwrap()),
        );
    }

    group.finish();
}

criterion_group!(
    linalg_dense_kernels,
    bench_dot,
    bench_matmul,
    bench_matmul_rhs_transposed,
    bench_matmul_rhs_strided_slice
);
criterion_main!(linalg_dense_kernels);
