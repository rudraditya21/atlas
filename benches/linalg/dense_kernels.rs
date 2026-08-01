use atlas_linalg::{dot, matmul};
use atlas_ndarray::NDArray;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

fn bench_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/dot");

    for &size in &[1_024usize, 16_384, 262_144] {
        let lhs = NDArray::from_shape_vec([size], vec![1.0_f64; size]).unwrap();
        let rhs = NDArray::from_shape_vec([size], vec![2.0_f64; size]).unwrap();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| dot(black_box(&lhs), black_box(&rhs)).unwrap())
        });
    }

    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/matmul_contiguous");

    for &size in &[32usize, 64, 128] {
        let lhs = NDArray::from_shape_vec([size, size], vec![1.0_f64; size * size]).unwrap();
        let rhs = NDArray::from_shape_vec([size, size], vec![2.0_f64; size * size]).unwrap();

        group.throughput(Throughput::Elements((size * size * size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| matmul(black_box(&lhs), black_box(&rhs)).unwrap())
        });
    }

    group.finish();
}

fn bench_matmul_rhs_transposed(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg/matmul_rhs_transposed");

    for &size in &[32usize, 64, 128] {
        let lhs = NDArray::from_shape_vec([size, size], vec![1.0_f64; size * size]).unwrap();
        let rhs_base = NDArray::from_shape_vec([size, size], vec![2.0_f64; size * size]).unwrap();

        group.throughput(Throughput::Elements((size * size * size) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| matmul(black_box(&lhs), black_box(rhs_base.view().transpose())).unwrap())
        });
    }

    group.finish();
}

criterion_group!(
    linalg_dense_kernels,
    bench_dot,
    bench_matmul,
    bench_matmul_rhs_transposed
);
criterion_main!(linalg_dense_kernels);
