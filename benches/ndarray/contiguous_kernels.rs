use atlas_ndarray::NDArray;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::time::Duration;

const VECTOR_BENCH_SIZES: [usize; 4] = [1 << 10, 1 << 14, 1 << 18, 1 << 20];
const MATRIX_BENCH_SIDES: [usize; 4] = [32, 128, 512, 1024];
const LARGE_INPUT_THRESHOLD: usize = 1 << 18;
const DEFAULT_SAMPLE_SIZE: usize = 100;
const LARGE_INPUT_SAMPLE_SIZE: usize = 50;
const DEFAULT_MEASUREMENT_SECS: u64 = 5;
const LARGE_INPUT_MEASUREMENT_SECS: u64 = 10;
const DEFAULT_WARM_UP_SECS: u64 = 3;

fn configure_group(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    elements: usize,
) {
    group.throughput(Throughput::Elements(elements as u64));
    group.warm_up_time(Duration::from_secs(DEFAULT_WARM_UP_SECS));

    if elements >= LARGE_INPUT_THRESHOLD {
        group.sample_size(LARGE_INPUT_SAMPLE_SIZE);
        group.measurement_time(Duration::from_secs(LARGE_INPUT_MEASUREMENT_SECS));
    } else {
        group.sample_size(DEFAULT_SAMPLE_SIZE);
        group.measurement_time(Duration::from_secs(DEFAULT_MEASUREMENT_SECS));
    }
}

fn filled_vector(len: usize, value: f64) -> NDArray<f64> {
    NDArray::from_vec(vec![len], vec![value; len]).unwrap()
}

fn filled_square_matrix(side: usize, value: f64) -> NDArray<f64> {
    NDArray::from_vec(vec![side, side], vec![value; side * side]).unwrap()
}

fn padded_square_source(side: usize, value: f64) -> NDArray<f64> {
    NDArray::from_vec(vec![side, side + 1], vec![value; side * (side + 1)]).unwrap()
}

fn bench_contiguous_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/add_contiguous");

    for size in VECTOR_BENCH_SIZES {
        configure_group(&mut group, size);
        let lhs = filled_vector(size, 1.0);
        let rhs = filled_vector(size, 2.0);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| black_box(lhs.add(black_box(&rhs)).unwrap()))
        });
    }

    group.finish();
}

fn bench_broadcast_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/add_broadcast");

    for side in MATRIX_BENCH_SIDES {
        let elements = side * side;
        configure_group(&mut group, elements);
        let lhs = filled_square_matrix(side, 1.0);
        let rhs = NDArray::from_vec(vec![1, side], vec![2.0; side]).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(elements), &elements, |b, _| {
            b.iter(|| black_box(lhs.add(black_box(&rhs)).unwrap()))
        });
    }

    group.finish();
}

fn bench_scalar_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/add_scalar");

    for size in VECTOR_BENCH_SIZES {
        configure_group(&mut group, size);
        let array = filled_vector(size, 1.0);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| black_box(array.add(black_box(2.0_f64))))
        });
    }

    group.finish();
}

fn bench_sum_layouts(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/sum_layouts");

    for side in MATRIX_BENCH_SIDES {
        let elements = side * side;
        configure_group(&mut group, elements);

        let contiguous = filled_square_matrix(side, 1.0);
        let transposed = contiguous.view().transpose();
        let padded = padded_square_source(side, 1.0);
        let sliced = padded.view().slice([0, 0], [side, side]).unwrap();

        group.bench_with_input(BenchmarkId::new("contiguous", elements), &elements, |b, _| {
            b.iter(|| black_box(contiguous.sum()))
        });
        group.bench_with_input(
            BenchmarkId::new("strided_transpose", elements),
            &elements,
            |b, _| b.iter(|| black_box(transposed.sum())),
        );
        group.bench_with_input(BenchmarkId::new("strided_slice", elements), &elements, |b, _| {
            b.iter(|| black_box(sliced.sum()))
        });
    }

    group.finish();
}

fn bench_mean_layouts(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray/mean_layouts");

    for side in MATRIX_BENCH_SIDES {
        let elements = side * side;
        configure_group(&mut group, elements);

        let contiguous = filled_square_matrix(side, 1.0);
        let transposed = contiguous.view().transpose();
        let padded = padded_square_source(side, 1.0);
        let sliced = padded.view().slice([0, 0], [side, side]).unwrap();

        group.bench_with_input(BenchmarkId::new("contiguous", elements), &elements, |b, _| {
            b.iter(|| black_box(contiguous.mean().unwrap()))
        });
        group.bench_with_input(
            BenchmarkId::new("strided_transpose", elements),
            &elements,
            |b, _| b.iter(|| black_box(transposed.mean().unwrap())),
        );
        group.bench_with_input(BenchmarkId::new("strided_slice", elements), &elements, |b, _| {
            b.iter(|| black_box(sliced.mean().unwrap()))
        });
    }

    group.finish();
}

criterion_group!(
    ndarray_contiguous_kernels,
    bench_contiguous_add,
    bench_broadcast_add,
    bench_scalar_add,
    bench_sum_layouts,
    bench_mean_layouts
);
criterion_main!(ndarray_contiguous_kernels);
